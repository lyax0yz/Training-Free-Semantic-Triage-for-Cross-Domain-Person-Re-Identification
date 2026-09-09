import argparse
import csv
import json
import math
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic'))
from arbitration import SymbolRegistry  # noqa: E402


ATTRIBUTES = ('upper_colour', 'lower_colour')
TOP_K = 20
DELTA = 0.02
CONFIDENCE_THRESHOLD = 0.25
LAMBDA_SEM = 0.05
CATEGORIES = ('corrected', 'harmful', 'reachable_not_corrected')


def font(size, bold=False):
    candidates = ([r'C:\Windows\Fonts\arialbd.ttf', r'C:\Windows\Fonts\segoeuib.ttf']
                  if bold else [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\segoeui.ttf'])
    for path in candidates:
        if osp.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = font(24, True)
LABEL_FONT = font(18, True)
TEXT_FONT = font(16)


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def cached_colours(item, registry):
    source = (item or {}).get('attributes', {})
    output = {}
    for name in ATTRIBUTES:
        raw = source.get(name, {})
        value = raw.get('value') if isinstance(raw, dict) else raw
        confidence = raw.get('confidence', 0.0) if isinstance(raw, dict) else 1.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        token = registry.normalize(name, value) if confidence >= CONFIDENCE_THRESHOLD else 'unknown'
        output[name] = {'value': token, 'raw_value': value or 'unknown',
                        'confidence': confidence}
    return output


def colour_adjustment(query, gallery):
    total = 0.0
    decisions = []
    for name in ATTRIBUTES:
        query_value = query[name]['value']
        gallery_value = gallery[name]['value']
        if 'unknown' in (query_value, gallery_value):
            contribution = 0.0
            decision = 'abstain'
        elif query_value == gallery_value:
            contribution = 0.5
            decision = 'match'
        else:
            contribution = -0.5
            decision = 'conflict'
        total += contribution
        decisions.append({'attribute': name, 'query_value': query_value,
                          'gallery_value': gallery_value, 'decision': decision,
                          'contribution': contribution})
    return total, decisions


def candidate_payload(index, scores, gallery_pids, gallery_camids,
                      gallery_paths, colours, adjustment, decisions):
    return {
        'gallery_index': int(index),
        'image_path': str(gallery_paths[index]),
        'pid': int(gallery_pids[index]),
        'camera_id': int(gallery_camids[index]),
        'visual_similarity': float(scores[index]),
        'semantic_adjustment': float(adjustment),
        'final_score': float(scores[index] + LAMBDA_SEM * adjustment),
        'attributes': colours,
        'decisions': decisions,
    }


def open_person(path, size=(250, 340)):
    image = Image.open(path).convert('RGB')
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, 'white')
    canvas.paste(image, ((size[0] - image.width) // 2,
                         (size[1] - image.height) // 2))
    return canvas


def draw_text(draw, x, y, lines, line_height=23):
    for line in lines:
        draw.text((x, y), line, fill='black', font=TEXT_FONT)
        y += line_height


def attribute_lines(attributes):
    return [
        '{}: {} (confidence={:.3f})'.format(
            name, attributes[name]['value'], attributes[name]['confidence'])
        for name in ATTRIBUTES
    ]


def decision_lines(candidate):
    return [
        '{}: {} ({:+.1f})'.format(
            item['attribute'], item['decision'], item['contribution'])
        for item in candidate['decisions']
    ]


def draw_column(canvas, x, label, colour, image_path, pid, camera_id,
                attributes, candidate=None):
    draw = ImageDraw.Draw(canvas)
    draw.text((x, 68), label, fill=colour, font=LABEL_FONT)
    canvas.paste(open_person(image_path), (x, 98))
    lines = ['PID={} camera={}'.format(pid, camera_id)]
    if candidate is not None:
        lines += [
            'visual={:.5f}'.format(candidate['visual_similarity']),
            'semantic={:+.4f}'.format(candidate['semantic_adjustment']),
            'final={:.5f}'.format(candidate['final_score']),
        ]
    lines += attribute_lines(attributes)
    if candidate is not None:
        lines += decision_lines(candidate)
    draw_text(draw, x, 450, lines)


def render_case(case, output_path):
    canvas = Image.new('RGB', (1240, 720), '#f3f3f3')
    draw = ImageDraw.Draw(canvas)
    title = ('SigLIP2 colour-only | {} | query {} | visual true-false gap={:+.5f}'
             .format(case['case_type'], case['query_index'],
                     case['true_minus_false_visual_gap']))
    subtitle = ('baseline Top-1 PID {}  ->  colour-only Top-1 PID {}'
                .format(case['baseline_top1_pid'], case['semantic_top1_pid']))
    draw.text((24, 12), title, fill='black', font=TITLE_FONT)
    draw.text((24, 42), subtitle, fill='black', font=TEXT_FONT)
    query = case['query']
    draw_column(canvas, 30, 'QUERY', '#1f4e79', query['image_path'], query['pid'],
                query['camera_id'], query['attributes'])
    true = case['true_candidate']
    draw_column(canvas, 430, 'TRUE CANDIDATE', '#16803c', true['image_path'],
                true['pid'], true['camera_id'], true['attributes'], true)
    false = case['competing_false_candidate']
    draw_column(canvas, 830, 'COMPETING FALSE', '#b22222', false['image_path'],
                false['pid'], false['camera_id'], false['attributes'], false)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    return canvas


def contact_sheet(images, output_path, cases_per_sheet=6):
    thumb_size = (620, 360)
    selected = images[:cases_per_sheet]
    rows = int(math.ceil(len(selected) / 2.0))
    sheet = Image.new('RGB', (1240, rows * 360), 'white')
    for index, source in enumerate(selected):
        thumb = source.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((index % 2) * 620, (index // 2) * 360))
    sheet.save(output_path)


def flatten_case(case):
    query = case['query']
    true = case['true_candidate']
    false = case['competing_false_candidate']
    row = {
        'case_type': case['case_type'], 'query_index': case['query_index'],
        'query_path': query['image_path'], 'query_pid': query['pid'],
        'query_camera_id': query['camera_id'],
        'baseline_top1_pid': case['baseline_top1_pid'],
        'semantic_top1_pid': case['semantic_top1_pid'],
        'true_path': true['image_path'], 'true_pid': true['pid'],
        'true_camera_id': true['camera_id'],
        'true_visual_similarity': true['visual_similarity'],
        'true_semantic_adjustment': true['semantic_adjustment'],
        'true_final_score': true['final_score'],
        'false_path': false['image_path'], 'false_pid': false['pid'],
        'false_camera_id': false['camera_id'],
        'false_visual_similarity': false['visual_similarity'],
        'false_semantic_adjustment': false['semantic_adjustment'],
        'false_final_score': false['final_score'],
        'true_minus_false_visual_gap': case['true_minus_false_visual_gap'],
    }
    for prefix, item in (('query', query), ('true', true), ('false', false)):
        for name in ATTRIBUTES:
            row['{}_{}_value'.format(prefix, name)] = item['attributes'][name]['value']
            row['{}_{}_confidence'.format(prefix, name)] = item['attributes'][name]['confidence']
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--attribute-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cases-per-sheet', type=int, default=6)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if osp.exists(args.output_dir) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite to replace files'.format(
            args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    with np.load(args.artifact, allow_pickle=False) as data:
        query_features = normalize(data['query_features'])
        gallery_features = normalize(data['gallery_features'])
        query_pids = data['query_pids']
        gallery_pids = data['gallery_pids']
        query_camids = data['query_camids']
        gallery_camids = data['gallery_camids']
        query_paths = data['query_impaths'].astype(str)
        gallery_paths = data['gallery_impaths'].astype(str)
    with open(args.attribute_cache, encoding='utf-8') as handle:
        cache_payload = json.load(handle)
    cache = cache_payload['images']
    registry = SymbolRegistry()

    cases = {category: [] for category in CATEGORIES}
    triggered = 0
    reachable_baseline_failures = 0
    for query_index in range(len(query_features)):
        scores = np.clip(query_features[query_index].dot(gallery_features.T), -1.0, 1.0)
        ranked = np.argsort(-scores, kind='mergesort')
        invalid = ((gallery_pids[ranked] == query_pids[query_index]) &
                   (gallery_camids[ranked] == query_camids[query_index]))
        valid = ranked[~invalid]
        top = valid[:TOP_K]
        ambiguity = top[scores[top] >= scores[top[0]] - DELTA]
        if len(ambiguity) <= 1:
            continue
        triggered += 1

        query_attributes = cached_colours(cache.get(query_paths[query_index]), registry)
        candidate_details = {}
        rescored = []
        for original_rank, gallery_index in enumerate(ambiguity):
            gallery_attributes = cached_colours(
                cache.get(gallery_paths[gallery_index]), registry)
            adjustment, decisions = colour_adjustment(
                query_attributes, gallery_attributes)
            candidate_details[int(gallery_index)] = candidate_payload(
                gallery_index, scores, gallery_pids, gallery_camids,
                gallery_paths, gallery_attributes, adjustment, decisions)
            rescored.append((
                -(scores[gallery_index] + LAMBDA_SEM * adjustment),
                -scores[gallery_index], original_rank, int(gallery_index)))
        rescored.sort()
        semantic_order = np.asarray([item[3] for item in rescored], dtype=np.int64)
        baseline_index = int(valid[0])
        semantic_index = int(semantic_order[0])
        baseline_correct = gallery_pids[baseline_index] == query_pids[query_index]
        semantic_correct = gallery_pids[semantic_index] == query_pids[query_index]
        true_indices = ambiguity[gallery_pids[ambiguity] == query_pids[query_index]]

        if not baseline_correct and len(true_indices):
            reachable_baseline_failures += 1
        if not baseline_correct and semantic_correct:
            category = 'corrected'
            true_index = semantic_index
            false_index = baseline_index
        elif baseline_correct and not semantic_correct:
            category = 'harmful'
            true_index = baseline_index
            false_index = semantic_index
        elif not baseline_correct and len(true_indices) and not semantic_correct:
            category = 'reachable_not_corrected'
            true_index = int(true_indices[0])
            false_index = semantic_index
        else:
            continue

        true_candidate = candidate_details[true_index]
        false_candidate = candidate_details[false_index]
        case = {
            'case_type': category,
            'query_index': int(query_index),
            'baseline_top1_pid': int(gallery_pids[baseline_index]),
            'semantic_top1_pid': int(gallery_pids[semantic_index]),
            'true_minus_false_visual_gap': float(
                scores[true_index] - scores[false_index]),
            'query': {
                'image_path': str(query_paths[query_index]),
                'pid': int(query_pids[query_index]),
                'camera_id': int(query_camids[query_index]),
                'attributes': query_attributes,
            },
            'true_candidate': true_candidate,
            'competing_false_candidate': false_candidate,
        }
        cases[category].append(case)

        if (query_index + 1) % 500 == 0:
            print('processed {}/{} queries'.format(
                query_index + 1, len(query_features)), flush=True)

    expected = {'corrected': 57, 'harmful': 36}
    for category, count in expected.items():
        if len(cases[category]) != count:
            raise AssertionError('{} count {} does not match saved experiment {}'.format(
                category, len(cases[category]), count))

    manifest = {'protocol': {
        'model': cache_payload.get('model'), 'attributes': list(ATTRIBUTES),
        'topk': TOP_K, 'delta': DELTA,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'lambda_sem': LAMBDA_SEM, 'fusion': 'signed_equal',
        'semantic_inference_rerun': False,
    }, 'triggered_query_count': triggered,
        'reachable_baseline_failure_count': reachable_baseline_failures,
        'categories': {}}
    csv_rows = []
    for category in CATEGORIES:
        category_dir = osp.join(args.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        rendered = []
        individual_paths = []
        for index, case in enumerate(cases[category]):
            path = osp.join(category_dir, '{:04d}_query_{:04d}.png'.format(
                index + 1, case['query_index']))
            rendered.append(render_case(case, path))
            individual_paths.append(path)
            csv_rows.append(flatten_case(case))
        contact_paths = []
        for start in range(0, len(rendered), args.cases_per_sheet):
            path = osp.join(args.output_dir, '{}_contact_{:03d}.png'.format(
                category, start // args.cases_per_sheet + 1))
            contact_sheet(rendered[start:start + args.cases_per_sheet], path,
                          args.cases_per_sheet)
            contact_paths.append(path)
        with open(osp.join(args.output_dir, category + '.json'), 'w',
                  encoding='utf-8') as handle:
            json.dump({'cases': cases[category]}, handle, indent=2)
        manifest['categories'][category] = {
            'count': len(cases[category]),
            'individual_pngs': individual_paths,
            'contact_sheets': contact_paths,
        }
        print('rendered {} {} cases'.format(
            len(cases[category]), category), flush=True)

    if csv_rows:
        with open(osp.join(args.output_dir, 'all_cases.csv'), 'w', newline='',
                  encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    with open(osp.join(args.output_dir, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print('saved {}'.format(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
