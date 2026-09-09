import argparse
import csv
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic'))
from scheme2 import Scheme2Config, compute_scheme2_adjustment  # noqa: E402


RANKS = (1, 5, 10)


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def new_metrics():
    return {'aps': [], 'cmc': {rank: [] for rank in RANKS}}


def add_metrics(state, order, query_pid, gallery_pids):
    matches = gallery_pids[order] == query_pid
    if not np.any(matches):
        return
    curve = np.minimum(1, matches.cumsum())
    for rank in RANKS:
        state['cmc'][rank].append(float(curve[rank - 1]))
    precision = matches.cumsum() / (np.arange(len(matches), dtype=np.float64) + 1.)
    state['aps'].append(float((precision * matches).sum() / matches.sum()))


def summarize(state):
    result = {'mAP': float(np.mean(state['aps']))}
    result.update({'Rank-{}'.format(rank): float(np.mean(state['cmc'][rank]))
                   for rank in RANKS})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--scheme2-cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transitions-csv', required=True)
    parser.add_argument('--source', default='market1501')
    parser.add_argument('--target', required=True)
    parser.add_argument('--distance-matrix',
                        help='Optional .npy query-gallery distance; similarity=1-distance.')
    parser.add_argument('--ranking-name', default='OSNet cosine')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if osp.exists(args.output) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite'.format(args.output))

    with open(args.config, encoding='utf-8') as handle:
        raw_config = json.load(handle)
    topk = int(raw_config.pop('candidate_topk'))
    delta = float(raw_config.pop('candidate_delta'))
    config = Scheme2Config.from_dict(raw_config)
    with open(args.scheme2_cache, encoding='utf-8') as handle:
        cache_payload = json.load(handle)
        cache = cache_payload['images']

    normalized_cache = {
        osp.normcase(osp.normpath(path)): item for path, item in cache.items()
    }

    def cached_item(path):
        return normalized_cache.get(osp.normcase(osp.normpath(path)))
    with np.load(args.artifact, allow_pickle=False) as artifact:
        query_features = normalize(artifact['query_features'])
        gallery_features = normalize(artifact['gallery_features'])
        query_pids, gallery_pids = artifact['query_pids'], artifact['gallery_pids']
        query_camids, gallery_camids = artifact['query_camids'], artifact['gallery_camids']
        query_paths = artifact['query_impaths'].astype(str)
        gallery_paths = artifact['gallery_impaths'].astype(str)
    distances = None
    if args.distance_matrix:
        distances = np.load(args.distance_matrix, mmap_mode='r')
        expected = (len(query_features), len(gallery_features))
        if distances.shape != expected:
            raise ValueError('Distance matrix shape {} != {}'.format(distances.shape, expected))

    baseline_state, semantic_state = new_metrics(), new_metrics()
    triggered = score_modified = ranking_changed = 0
    wrong_to_correct = correct_to_wrong = 0
    ambiguity_sizes = []
    transitions = []
    accepted_count = possible_count = 0
    agreement = {'true': [0, 0], 'false': [0, 0]}
    missing_paths = set()

    for query_index in range(len(query_features)):
        scores = (1. - np.asarray(distances[query_index], dtype=np.float32)
                  if distances is not None else
                  np.clip(query_features[query_index].dot(gallery_features.T), -1., 1.))
        visual_order = np.argsort(-scores, kind='mergesort')
        invalid = ((gallery_pids[visual_order] == query_pids[query_index]) &
                   (gallery_camids[visual_order] == query_camids[query_index]))
        valid = visual_order[~invalid]
        add_metrics(baseline_state, valid, query_pids[query_index], gallery_pids)
        top = valid[:topk]
        ambiguity = top[scores[top] >= scores[top[0]] - delta]
        ambiguity_sizes.append(len(ambiguity))
        baseline_correct = gallery_pids[valid[0]] == query_pids[query_index]

        if len(ambiguity) == 1:
            add_metrics(semantic_state, valid, query_pids[query_index], gallery_pids)
            continue
        triggered += 1
        query_item = cached_item(query_paths[query_index])
        if query_item is None:
            missing_paths.add(query_paths[query_index])
            continue
        query_attributes = query_item['aggregated_attributes']
        rescored = []
        any_nonzero = False
        for original_rank, gallery_index_value in enumerate(ambiguity):
            gallery_index = int(gallery_index_value)
            gallery_item = cached_item(gallery_paths[gallery_index])
            if gallery_item is None:
                missing_paths.add(gallery_paths[gallery_index])
                continue
            gallery_attributes = gallery_item['aggregated_attributes']
            detail = compute_scheme2_adjustment(
                query_attributes, gallery_attributes, config,
                use_uncertainty_gate=False, use_quality_mask=False)
            adjustment = float(detail['effective_semantic_adjustment'])
            any_nonzero = any_nonzero or abs(adjustment) > 1e-12
            final_score = float(scores[gallery_index] + config.lambda_sem * adjustment)
            rescored.append((-final_score, -float(scores[gallery_index]),
                             original_rank, gallery_index))
            accepted_count += detail['accepted_attribute_count']
            possible_count += len(config.attribute_weights)
            pair_kind = 'true' if gallery_pids[gallery_index] == query_pids[query_index] else 'false'
            for decision in detail['decisions']:
                if decision['accepted']:
                    agreement[pair_kind][1] += 1
                    agreement[pair_kind][0] += int(decision['reason'] == 'match')
        if missing_paths:
            break
        rescored.sort()
        semantic_prefix = np.asarray([item[3] for item in rescored], dtype=np.int64)
        semantic_order = np.concatenate((semantic_prefix, valid[len(ambiguity):]))
        add_metrics(semantic_state, semantic_order, query_pids[query_index], gallery_pids)
        score_modified += int(any_nonzero)
        ranking_changed += int(not np.array_equal(semantic_prefix, ambiguity))
        semantic_correct = gallery_pids[semantic_order[0]] == query_pids[query_index]
        if not baseline_correct and semantic_correct:
            wrong_to_correct += 1; transition = 'wrong_to_correct'
        elif baseline_correct and not semantic_correct:
            correct_to_wrong += 1; transition = 'correct_to_wrong'
        else:
            transition = None
        if transition:
            transitions.append({
                'query_index': query_index, 'query_path': query_paths[query_index],
                'query_pid': int(query_pids[query_index]), 'transition': transition,
                'baseline_top1_path': gallery_paths[valid[0]],
                'baseline_top1_pid': int(gallery_pids[valid[0]]),
                'semantic_top1_path': gallery_paths[semantic_order[0]],
                'semantic_top1_pid': int(gallery_pids[semantic_order[0]])})
        if (query_index + 1) % 250 == 0:
            print('Processed {}/{} queries'.format(query_index + 1, len(query_features)), flush=True)

    if missing_paths:
        raise ValueError('Semantic cache missing {} required path(s); first={}'.format(
            len(missing_paths), sorted(missing_paths)[0]))
    baseline, semantic = summarize(baseline_state), summarize(semantic_state)
    report = {
        'experiment': 'Frozen Scheme 2 configuration D external validation',
        'source': args.source, 'target': args.target,
        'ranking_source': args.ranking_name,
        'ranking_score_definition': ('1 - saved distance' if distances is not None else
                                     'cosine similarity of L2-normalized OSNet features'),
        'fixed_configuration': {
            'topk': topk, 'delta': delta, 'lambda_sem': config.lambda_sem,
            'attributes': list(config.attribute_weights),
            'attribute_weights': config.attribute_weights,
            'scheme2_variant': 'D: SymbolRegistry + deterministic multi-view voting; no gate/quality mask'},
        'baseline': baseline, 'scheme2_D': semantic,
        'change_from_baseline': {key: semantic[key] - baseline[key] for key in semantic},
        'candidate_statistics': {
            'query_count': len(query_features), 'triggered_query_count': triggered,
            'triggered_query_rate': triggered / float(len(query_features)),
            'average_ambiguity_size': float(np.mean(ambiguity_sizes))},
        'semantic_statistics': {
            'semantically_score_modified_query_count': score_modified,
            'ranking_changed_query_count': ranking_changed,
            'wrong_to_correct_count': wrong_to_correct,
            'correct_to_wrong_count': correct_to_wrong,
            'net_rank1_correction_count': wrong_to_correct - correct_to_wrong,
            'average_accepted_attributes_per_pair': accepted_count / float(max(1, possible_count // len(config.attribute_weights))),
            'semantic_abstention_rate': 1. - accepted_count / float(max(1, possible_count)),
            'true_match_attribute_agreement_rate': agreement['true'][0] / float(max(1, agreement['true'][1])),
            'false_match_attribute_agreement_rate': agreement['false'][0] / float(max(1, agreement['false'][1]))},
        'inference_label_policy': 'No PID/camera labels used in candidate scoring or semantic fusion.'}
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    fields = ['query_index', 'query_path', 'query_pid', 'transition',
              'baseline_top1_path', 'baseline_top1_pid',
              'semantic_top1_path', 'semantic_top1_pid']
    with open(args.transitions_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(transitions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
