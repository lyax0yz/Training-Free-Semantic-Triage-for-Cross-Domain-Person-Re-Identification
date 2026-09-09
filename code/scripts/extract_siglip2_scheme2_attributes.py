import argparse
import csv
import json
import math
import os
import os.path as osp
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic'))
from scheme2 import (ATTRIBUTE_NAMES, PedestrianSymbolRegistry,
                     aggregate_image_predictions)  # noqa: E402


MODEL_ID = 'google/siglip2-base-patch16-224'

RAW_VOCABULARIES = {
    'gender': ('man', 'woman'),
    'build': ('slim', 'average', 'stocky'),
    'age': ('young adult', 'middle aged adult', 'older adult'),
    'upper_colour': ('black', 'white', 'gray', 'red', 'orange', 'yellow',
                     'green', 'blue', 'purple', 'pink', 'brown'),
    'upper_type': ('t-shirt', 'shirt', 'jacket', 'coat', 'hoodie', 'sweater',
                   'dress', 'other'),
    'lower_colour': ('black', 'white', 'gray', 'red', 'orange', 'yellow',
                     'green', 'blue', 'purple', 'pink', 'brown'),
    'lower_type': ('trousers', 'jeans', 'shorts', 'skirt', 'dress', 'leggings',
                   'other'),
    'backpack': ('present', 'absent'),
    'handbag': ('present', 'absent'),
    'hat': ('present', 'absent'),
}

RELEVANT_VIEWS = {
    'gender': ('original', 'center_crop', 'upper_body', 'horizontal_flip'),
    'build': ('original', 'center_crop', 'horizontal_flip'),
    'age': ('original', 'center_crop', 'upper_body', 'horizontal_flip'),
    'upper_colour': ('original', 'center_crop', 'upper_body', 'horizontal_flip'),
    'upper_type': ('original', 'center_crop', 'upper_body', 'horizontal_flip'),
    'lower_colour': ('original', 'center_crop', 'lower_body', 'horizontal_flip'),
    'lower_type': ('original', 'center_crop', 'lower_body', 'horizontal_flip'),
    'backpack': ('original', 'center_crop', 'horizontal_flip'),
    'handbag': ('original', 'center_crop', 'horizontal_flip'),
    'hat': ('original', 'center_crop', 'upper_body', 'horizontal_flip'),
}


def attribute_prompt(attribute, value):
    unknown = value == 'unknown'
    if attribute == 'gender':
        return ('a full-body pedestrian whose perceived gender is unclear' if unknown else
                'a full-body photo of a {} pedestrian'.format(value))
    if attribute == 'build':
        return ('a full-body pedestrian whose body build is unclear' if unknown else
                'a full-body photo of a pedestrian with a {} body build'.format(value))
    if attribute == 'age':
        return ('a full-body pedestrian whose age group is unclear' if unknown else
                'a full-body photo of a {} pedestrian'.format(value))
    if attribute == 'upper_colour':
        return ('a full-body pedestrian whose upper clothing colour is unclear' if unknown else
                'a full-body photo of a pedestrian wearing {} upper-body clothing'.format(value))
    if attribute == 'lower_colour':
        return ('a full-body pedestrian whose lower clothing colour is unclear' if unknown else
                'a full-body photo of a pedestrian wearing {} lower-body clothing'.format(value))
    if attribute == 'upper_type':
        return ('a full-body pedestrian whose upper clothing type is unclear' if unknown else
                'a full-body photo of a pedestrian wearing a {} as upper-body clothing'.format(value))
    if attribute == 'lower_type':
        return ('a full-body pedestrian whose lower clothing type is unclear' if unknown else
                'a full-body photo of a pedestrian wearing {} as lower-body clothing'.format(value))
    noun = attribute
    if value == 'present':
        return 'a full-body photo of a pedestrian wearing or carrying a {}'.format(noun)
    if value == 'absent':
        return 'a full-body photo of a pedestrian without a {}'.format(noun)
    return 'a full-body pedestrian image where the {} is not visible or cannot be determined'.format(noun)


def deterministic_views(image):
    from PIL import ImageOps
    width, height = image.size
    center = image.crop((int(.05 * width), int(.03 * height),
                         max(int(.95 * width), 1), max(int(.97 * height), 1)))
    upper = image.crop((0, 0, width, max(int(.68 * height), 1)))
    lower = image.crop((0, min(int(.32 * height), height - 1), width, height))
    return {
        'original': image,
        'center_crop': center,
        'upper_body': upper,
        'lower_body': lower,
        'horizontal_flip': ImageOps.mirror(image),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--inspection-csv', required=True)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--hf-cache', default='artifacts/siglip2-hf-cache')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of source images; five views are encoded per image.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='Inference device. Auto selects CUDA when it is available.')
    parser.add_argument('--precision', choices=('auto', 'float32', 'float16', 'bfloat16'),
                        default='auto',
                        help='Model precision. Auto uses float16 on CUDA and float32 on CPU.')
    parser.add_argument('--torch-threads', type=int, default=0,
                        help='CPU intra-op threads (0 keeps the PyTorch default).')
    parser.add_argument('--min-agreement', type=float, default=.5)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--randomize-missing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Split the deterministic path list into disjoint extraction shards.')
    parser.add_argument('--shard-index', type=int, default=0,
                        help='Zero-based shard index; merge shard caches after extraction.')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if not 0. <= args.min_agreement <= 1.:
        raise ValueError('--min-agreement must be in [0, 1]')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require num_shards >= 1 and 0 <= shard_index < num_shards')
    if osp.exists(args.cache) and not args.resume:
        raise FileExistsError('{} exists; pass --resume'.format(args.cache))

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    if args.torch_threads:
        if args.torch_threads < 1:
            raise ValueError('--torch-threads must be positive')
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    with open(args.candidates, encoding='utf-8') as handle:
        candidates = json.load(handle)
    existing = {}
    if args.resume and osp.exists(args.cache):
        with open(args.cache, encoding='utf-8') as handle:
            existing = json.load(handle).get('images', {})

    def complete(item):
        aggregated = item.get('aggregated_attributes', item.get('attributes', {}))
        return all(name in aggregated and
                   math.isfinite(float(aggregated[name].get('confidence', float('nan'))))
                   for name in ATTRIBUTE_NAMES)

    paths = [path for path in candidates['unique_image_paths']
             if path not in existing or not complete(existing[path])]
    if args.randomize_missing:
        random.Random(args.seed).shuffle(paths)
    if args.max_images:
        paths = paths[:args.max_images]
    paths = paths[args.shard_index::args.num_shards]
    if not paths:
        print('Cache already contains all requested images; nothing to extract.', flush=True)
        return

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but torch.cuda.is_available() is False')
    precision = args.precision
    if precision == 'auto':
        precision = 'float16' if device == 'cuda' else 'float32'
    if device == 'cpu' and precision == 'float16':
        raise ValueError('float16 CPU inference is unsupported; use float32 or bfloat16')
    model_dtype = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[precision]
    hf_cache = osp.abspath(args.hf_cache)
    processor = AutoProcessor.from_pretrained(
        args.model_id, cache_dir=hf_cache, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(
        args.model_id, cache_dir=hf_cache, local_files_only=args.local_files_only
    ).to(device=device, dtype=model_dtype).eval()
    registry = PedestrianSymbolRegistry()

    prompt_offsets = {}
    all_prompts = []
    for attribute in ATTRIBUTE_NAMES:
        start = len(all_prompts)
        all_prompts.extend(attribute_prompt(attribute, value)
                           for value in RAW_VOCABULARIES[attribute])
        prompt_offsets[attribute] = (start, len(all_prompts))

    text_inputs = processor(text=all_prompts, padding='max_length', return_tensors='pt')
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()
                   if key in ('input_ids', 'attention_mask', 'position_ids')}
    with torch.inference_mode():
        text_output = model.get_text_features(**text_inputs)
        text_features = text_output.pooler_output if hasattr(text_output, 'pooler_output') else text_output
        text_features = functional.normalize(text_features.float(), dim=-1)
        logit_scale = model.logit_scale.float().exp()
        logit_bias = model.logit_bias.float()

    results = dict(existing)
    errors = []

    def payload():
        return {
            'schema_version': 2,
            'model': args.model_id,
            'backend': 'transformers.AutoModel/AutoProcessor (SigLIP 2)',
            'candidate_topk': candidates.get('topk'),
            'candidate_delta': candidates.get('delta'),
            'shard': {'index': args.shard_index, 'count': args.num_shards,
                      'path_count': len(paths)},
            'attributes': list(ATTRIBUTE_NAMES),
            'raw_vocabularies': {key: list(value) for key, value in RAW_VOCABULARIES.items()},
            'normalized_vocabularies': {
                key: list(PedestrianSymbolRegistry.VOCABULARIES[key])
                for key in ATTRIBUTE_NAMES},
            'prompts': {attribute: [attribute_prompt(attribute, value)
                                    for value in RAW_VOCABULARIES[attribute]]
                        for attribute in ATTRIBUTE_NAMES},
            'views': ['original', 'center_crop', 'upper_body', 'lower_body', 'horizontal_flip'],
            'relevant_views': {key: list(value) for key, value in RELEVANT_VIEWS.items()},
            'aggregation': {'method': 'confidence_weighted_vote',
                            'minimum_vote_agreement': args.min_agreement},
            'runtime': {'torch_threads': (args.torch_threads or torch.get_num_threads()),
                        'batch_size': args.batch_size, 'device': device,
                        'precision': precision},
            'images': results, 'errors': errors,
        }

    def write_cache():
        os.makedirs(osp.dirname(osp.abspath(args.cache)), exist_ok=True)
        temporary = args.cache + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload(), handle, separators=(',', ':'))
        os.replace(temporary, args.cache)

    view_order = ('original', 'center_crop', 'upper_body', 'lower_body', 'horizontal_flip')
    last_saved = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        source_paths, flat_views = [], []
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                views = deterministic_views(image)
                flat_views.extend(views[name] for name in view_order)
                source_paths.append(path)
            except Exception as exc:
                errors.append({'image_path': path, 'error': str(exc)})
        if source_paths:
            image_inputs = processor(images=flat_views, return_tensors='pt')
            image_inputs = {
                key: (value.to(device=device, dtype=model_dtype)
                      if key == 'pixel_values' else value.to(device))
                for key, value in image_inputs.items()
                if key in ('pixel_values', 'pixel_attention_mask', 'spatial_shapes')}
            with torch.inference_mode():
                image_output = model.get_image_features(**image_inputs)
                image_features = (image_output.pooler_output if hasattr(image_output, 'pooler_output')
                                  else image_output)
                image_features = functional.normalize(image_features.float(), dim=-1)
                all_logits = image_features @ text_features.T
                all_logits = all_logits * logit_scale + logit_bias
            for image_index, path in enumerate(source_paths):
                raw_views = {}
                for view_index, view_name in enumerate(view_order):
                    row = all_logits[image_index * len(view_order) + view_index]
                    raw_views[view_name] = {}
                    for attribute in ATTRIBUTE_NAMES:
                        offset_start, offset_end = prompt_offsets[attribute]
                        probabilities = torch.softmax(row[offset_start:offset_end], dim=0)
                        best = int(probabilities.argmax())
                        raw_views[view_name][attribute] = {
                            'value': RAW_VOCABULARIES[attribute][best],
                            'confidence': float(probabilities[best].cpu())}
                normalized, aggregated = aggregate_image_predictions(
                    raw_views, RELEVANT_VIEWS, registry, args.min_agreement)
                results[path] = {
                    'image_path': path, 'raw_views': raw_views,
                    'normalized_views': normalized,
                    'aggregated_attributes': aggregated,
                    'attributes': aggregated}
        processed = min(start + args.batch_size, len(paths))
        if args.save_every and (processed - last_saved >= args.save_every or processed == len(paths)):
            write_cache()
            last_saved = processed
        print('Processed {}/{} missing source images'.format(processed, len(paths)), flush=True)

    if len(errors) > max(1, int(.05 * max(1, len(paths)))):
        raise RuntimeError('Aborted: too many unreadable images ({}/{})'.format(len(errors), len(paths)))
    write_cache()

    random.seed(args.seed)
    sample = random.sample(list(results.values()), min(args.sample_size, len(results)))
    fields = ['image_path']
    for attribute in ATTRIBUTE_NAMES:
        fields.extend((attribute, attribute + '_confidence', attribute + '_vote_agreement',
                       attribute + '_num_valid_views'))
    with open(args.inspection_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sample:
            row = {'image_path': item['image_path']}
            for attribute in ATTRIBUTE_NAMES:
                value = item['aggregated_attributes'][attribute]
                row[attribute] = value['value']
                row[attribute + '_confidence'] = value['confidence']
                row[attribute + '_vote_agreement'] = value['vote_agreement']
                row[attribute + '_num_valid_views'] = value['num_valid_views']
            writer.writerow(row)
    print('Saved cache {} and inspection {}'.format(args.cache, args.inspection_csv), flush=True)


if __name__ == '__main__':
    main()
