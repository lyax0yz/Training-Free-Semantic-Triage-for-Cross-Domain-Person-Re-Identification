import argparse
import csv
import json
import math
import os
import os.path as osp
import random

from code.scripts.pedestrian_attributes_draft import ATTRIBUTE_PROMPTS, prompt


MODEL_ID = 'google/siglip2-base-patch16-224'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--inspection-csv', required=True)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--hf-cache', default='artifacts/siglip2-hf-cache')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Use 1 on memory-limited Windows machines.')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Process at most this many missing images (0 means all).')
    parser.add_argument('--randomize-missing', action='store_true',
                        help='Randomly sample missing paths before --max-images; useful for a representative pilot.')
    parser.add_argument('--min-confidence', type=float, default=.25)
    parser.add_argument('--attributes', nargs='+', default=['upper_colour', 'lower_colour'],
                        help='Defaults to the current best colour-only representation.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if any(name not in ATTRIBUTE_PROMPTS for name in args.attributes):
        raise ValueError('Unknown attribute; choose from {}'.format(', '.join(ATTRIBUTE_PROMPTS)))
    if osp.exists(args.cache) and not args.resume:
        raise FileExistsError('{} exists; pass --resume to fill only missing images'.format(args.cache))

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    with open(args.candidates, encoding='utf-8') as handle:
        candidates = json.load(handle)
    existing = {}
    if args.resume and osp.exists(args.cache):
        with open(args.cache, encoding='utf-8') as handle:
            existing = json.load(handle).get('images', {})

    def complete(item):
        attributes = item.get('attributes', {})
        return all(name in attributes and math.isfinite(float(attributes[name].get('confidence', float('nan'))))
                   for name in args.attributes)

    paths = [path for path in candidates['unique_image_paths']
             if path not in existing or not complete(existing[path])]
    if args.randomize_missing:
        random.Random(args.seed).shuffle(paths)
    if args.max_images:
        paths = paths[:args.max_images]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf_cache = osp.abspath(args.hf_cache)
    os.makedirs(hf_cache, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=hf_cache)
    model = AutoModel.from_pretrained(args.model_id, cache_dir=hf_cache).to(device).eval()
    text_prompts = {name: [prompt(name, value) for value in ATTRIBUTE_PROMPTS[name]]
                    for name in args.attributes}
    errors, results = [], dict(existing)

    def write_cache():
        known = sum(value['value'] != 'unknown' for item in results.values()
                    for name, value in item['attributes'].items() if name in args.attributes)
        payload = {
            'model': args.model_id,
            'backend': 'transformers.AutoModel/AutoProcessor (SigLIP 2)',
            'min_confidence': args.min_confidence, 'extracted_attributes': args.attributes,
            'candidate_topk': candidates.get('topk'), 'candidate_delta': candidates.get('delta'),
            'images': results, 'errors': errors,
            'known_attribute_rate': known / max(1, len(results) * len(args.attributes))}
        os.makedirs(osp.dirname(osp.abspath(args.cache)), exist_ok=True)
        temporary = args.cache + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, args.cache)

    for start in range(0, len(paths), args.batch_size):
        batch_paths, images = paths[start:start + args.batch_size], []
        usable = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert('RGB'))
                usable.append(path)
            except Exception as exc:
                errors.append({'image_path': path, 'error': str(exc)})
        if images:
            attributes_by_image = [dict(results.get(path, {}).get('attributes', {})) for path in usable]
            with torch.inference_mode():
                for attribute in args.attributes:
                    values = ATTRIBUTE_PROMPTS[attribute]
                    inputs = processor(text=text_prompts[attribute], images=images,
                                       padding='max_length', return_tensors='pt').to(device)
                    logits = model(**inputs).logits_per_image.float()
                    
                    probabilities = torch.softmax(logits, dim=-1).cpu()
                    if not torch.isfinite(probabilities).all():
                        raise RuntimeError('SigLIP 2 produced non-finite probabilities for {}'.format(attribute))
                    for index, probability in enumerate(probabilities):
                        best = int(probability.argmax())
                        confidence = float(probability[best])
                        value = values[best] if confidence >= args.min_confidence else 'unknown'
                        attributes_by_image[index][attribute] = {'value': value, 'confidence': confidence}
            for path, attributes in zip(usable, attributes_by_image):
                results[path] = {'image_path': path, 'attributes': attributes}
        processed = min(start + args.batch_size, len(paths))
        if args.save_every and (processed % args.save_every == 0 or processed == len(paths)):
            write_cache()
        print('Processed {}/{} missing images'.format(processed, len(paths)), flush=True)

    if len(errors) > max(1, int(.05 * len(paths))):
        raise RuntimeError('Aborted: too many unreadable images ({}/{})'.format(len(errors), len(paths)))
    write_cache()
    random.seed(args.seed)
    sample = random.sample(list(results.values()), min(args.sample_size, len(results)))
    with open(args.inspection_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path'] + list(args.attributes))
        for item in sample:
            writer.writerow([item['image_path']] + [item['attributes'].get(name, {}).get('value', 'unknown') for name in args.attributes])
    print('Saved {} and {}'.format(args.cache, args.inspection_csv))


if __name__ == '__main__':
    main()
