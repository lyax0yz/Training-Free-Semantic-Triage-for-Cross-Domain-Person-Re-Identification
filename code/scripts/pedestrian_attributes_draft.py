import argparse
import csv
import json
import math
import os
import os.path as osp
import random


ATTRIBUTE_PROMPTS = {
    'upper_colour': ['black', 'white', 'gray', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown'],
    'lower_colour': ['black', 'white', 'gray', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown'],
    'upper_type': ['tshirt', 'shirt', 'jacket', 'coat', 'hoodie', 'sweater', 'dress', 'other'],
    'lower_type': ['trousers', 'jeans', 'shorts', 'skirt', 'dress', 'leggings', 'unknown'],
    'bag': ['backpack', 'shoulder_bag', 'handbag', 'no_bag', 'unknown']
}


def prompt(attribute, value):
    templates = {
        'upper_colour': 'a full-body photo of a pedestrian wearing a {} upper-body garment',
        'lower_colour': 'a full-body photo of a pedestrian wearing {} lower-body clothing',
        'upper_type': 'a full-body photo of a pedestrian wearing a {} as upper-body clothing',
        'lower_type': 'a full-body photo of a pedestrian wearing {} as lower-body clothing',
        'bag': {'backpack': 'a full-body photo of a pedestrian carrying a backpack',
                'shoulder_bag': 'a full-body photo of a pedestrian carrying a shoulder bag',
                'handbag': 'a full-body photo of a pedestrian carrying a handbag',
                'no_bag': 'a full-body photo of a pedestrian carrying no bag',
                'unknown': 'a full-body pedestrian image where bag presence cannot be determined'}
    }
    return templates[attribute][value] if isinstance(templates[attribute], dict) else templates[attribute].format(value.replace('_', ' '))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True, help='candidate_selection JSON')
    p.add_argument('--cache', required=True)
    p.add_argument('--inspection-csv', required=True)
    p.add_argument('--model', default='ViT-B-32')
    p.add_argument('--pretrained', default='openai')
    p.add_argument('--clip-cache', default='artifacts/clip-cache',
                   help='Project-writable cache for OpenAI CLIP weights')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--min-confidence', type=float, default=.25)
    p.add_argument('--sample-size', type=int, default=20)
    p.add_argument('--max-images', type=int, default=0, help='Extract at most this many missing images (0=all)')
    p.add_argument('--randomize-missing', action='store_true',
                   help='Randomly choose missing images before applying --max-images; useful for the safety pilot')
    p.add_argument('--resume', action='store_true', help='Reuse an existing cache and process only missing images')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save-every', type=int, default=200,
                   help='Persist valid progress every N processed images (0=only at end)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if osp.exists(args.cache) and not args.overwrite and not args.resume:
        raise FileExistsError('{} exists; pass --overwrite'.format(args.cache))
    try:
        import open_clip
        clip_backend = 'open_clip'
    except ImportError:
        try:
            import clip
            clip_backend = 'openai_clip'
        except ImportError as exc:
            raise RuntimeError('Install OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git') from exc
    from PIL import Image
    import torch
    with open(args.candidates, encoding='utf-8') as f:
        candidate_info = json.load(f)
    existing = {}
    if args.resume and osp.exists(args.cache):
        with open(args.cache, encoding='utf-8') as f:
            existing = json.load(f).get('images', {})
    def usable_cached_result(item):
        attributes = item.get('attributes', {})
        return all(name in attributes and math.isfinite(float(attributes[name].get('confidence', float('nan'))))
                   for name in ATTRIBUTE_PROMPTS)

    paths = [path for path in candidate_info['unique_image_paths']
             if path not in existing or not usable_cached_result(existing[path])]
    if args.randomize_missing:
        random.Random(args.seed).shuffle(paths)
    if args.max_images:
        paths = paths[:args.max_images]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if clip_backend == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(args.model)
    else:
        clip_cache = osp.abspath(args.clip_cache)
        os.makedirs(clip_cache, exist_ok=True)
        
        try:
            import certifi
            import ssl
            os.environ.setdefault('SSL_CERT_FILE', certifi.where())
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        model, preprocess = clip.load('ViT-B/32', device=device, download_root=clip_cache)
        tokenizer = clip.tokenize

    model = model.float()
    model.eval()
    text_features = {}
    with torch.no_grad():
        for attribute, values in ATTRIBUTE_PROMPTS.items():
            text = tokenizer([prompt(attribute, v) for v in values]).to(device)
            text_features[attribute] = model.encode_text(text).float()
            text_features[attribute] /= text_features[attribute].norm(dim=-1, keepdim=True)
            if not torch.isfinite(text_features[attribute]).all():
                raise RuntimeError('CLIP produced non-finite text features for {}'.format(attribute))
    results, errors = dict(existing), []

    def write_cache():
        known = sum(v['value'] != 'unknown' for item in results.values()
                    for v in item['attributes'].values())
        known_rate = known / max(1, len(results) * len(ATTRIBUTE_PROMPTS))
        payload = {'model': args.model, 'pretrained': args.pretrained,
                   'min_confidence': args.min_confidence, 'images': results,
                   'errors': errors, 'known_attribute_rate': known_rate}
        os.makedirs(osp.dirname(osp.abspath(args.cache)), exist_ok=True)
        temp_cache = args.cache + '.tmp'
        with open(temp_cache, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_cache, args.cache)
        return known_rate

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        images, usable = [], []
        for path in batch_paths:
            try:
                images.append(preprocess(Image.open(path).convert('RGB')))
                usable.append(path)
            except Exception as exc:
                errors.append({'image_path': path, 'error': str(exc)})
        if images:
            with torch.no_grad():
                features = model.encode_image(torch.stack(images).to(device)).float()
                features /= features.norm(dim=-1, keepdim=True)
                if not torch.isfinite(features).all():
                    raise RuntimeError('CLIP produced non-finite image features; aborting rather than saving invalid attributes')
                for idx, path in enumerate(usable):
                    attrs = dict(results.get(path, {}).get('attributes', {}))
                    for attribute, values in ATTRIBUTE_PROMPTS.items():
                        if attribute in attrs and math.isfinite(float(attrs[attribute].get('confidence', float('nan')))):
                            continue
                        probs = (100.0 * features[idx:idx + 1] @ text_features[attribute].T).softmax(dim=-1)[0]
                        if not torch.isfinite(probs).all():
                            raise RuntimeError('CLIP produced non-finite attribute probabilities; aborting')
                        best = int(probs.argmax())
                        value, confidence = values[best], float(probs[best])
                        if confidence < args.min_confidence:
                            value = 'unknown'
                        attrs[attribute] = {'value': value, 'confidence': confidence}
                    results[path] = {'image_path': path, 'attributes': attrs}
        processed = min(start + args.batch_size, len(paths))
        if args.save_every and (processed % args.save_every == 0 or processed == len(paths)):
            write_cache()
        print('Processed {}/{} images'.format(processed, len(paths)))
    if len(errors) > max(1, int(.05 * len(paths))):
        raise RuntimeError('Attribute extraction aborted: too many unreadable images ({}/{})'.format(len(errors), len(paths)))
    known = sum(v['value'] != 'unknown' for item in results.values() for v in item['attributes'].values())
    known_rate = known / max(1, len(results) * len(ATTRIBUTE_PROMPTS))
    if known_rate < .20:
        raise RuntimeError('Attribute extraction aborted: known attribute rate {:.1%} is nonsensical'.format(known_rate))
    known_rate = write_cache()
    random.seed(args.seed)
    sample = random.sample(list(results.values()), min(args.sample_size, len(results)))
    with open(args.inspection_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image_path'] + list(ATTRIBUTE_PROMPTS))
        for item in sample:
            w.writerow([item['image_path']] + [item['attributes'][a]['value'] for a in ATTRIBUTE_PROMPTS])
    print('Saved cache {}, manual inspection CSV {}, known rate {:.1%}'.format(args.cache, args.inspection_csv, known_rate))


if __name__ == '__main__':
    main()
