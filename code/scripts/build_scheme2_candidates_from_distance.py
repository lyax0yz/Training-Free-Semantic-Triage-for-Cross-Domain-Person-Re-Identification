import argparse
import json
import os.path as osp

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--distance-matrix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--delta', type=float, default=.02)
    parser.add_argument('--ranking-name', required=True)
    args = parser.parse_args()
    distances = np.load(args.distance_matrix, mmap_mode='r')
    with np.load(args.artifact, allow_pickle=False) as data:
        q_pids, g_pids = data['query_pids'], data['gallery_pids']
        q_camids, g_camids = data['query_camids'], data['gallery_camids']
        q_paths, g_paths = data['query_impaths'].astype(str), data['gallery_impaths'].astype(str)
    if distances.shape != (len(q_paths), len(g_paths)):
        raise ValueError('Distance matrix shape mismatch')
    queries, unique_paths = [], set()
    triggered = 0
    for qi in range(len(q_paths)):
        scores = 1. - np.asarray(distances[qi], dtype=np.float32)
        order = np.argsort(-scores, kind='mergesort')
        invalid = ((g_pids[order] == q_pids[qi]) & (g_camids[order] == q_camids[qi]))
        valid = order[~invalid]
        top = valid[:args.topk]
        ambiguity = top[scores[top] >= scores[top[0]] - args.delta]
        is_triggered = len(ambiguity) > 1
        if is_triggered:
            triggered += 1
            unique_paths.add(q_paths[qi])
            unique_paths.update(g_paths[index] for index in ambiguity)
        queries.append({
            'query_index': qi, 'query_image_path': q_paths[qi],
            'triggered': is_triggered,
            'top1_score': float(scores[top[0]]),
            'candidate_gallery_indices': [int(index) for index in ambiguity],
            'candidate_gallery_paths': [g_paths[index] for index in ambiguity],
            'candidate_scores': [float(scores[index]) for index in ambiguity]})
        if (qi + 1) % 500 == 0:
            print('Candidate rows {}/{}'.format(qi + 1, len(q_paths)), flush=True)
    payload = {
        'artifact': osp.abspath(args.artifact),
        'ranking_source': args.ranking_name,
        'score_definition': '1 - saved distance',
        'topk': args.topk, 'delta': args.delta,
        'semantic_trigger_rule': 'ambiguity_size > 1',
        'triggered_query_count': triggered,
        'queries': queries, 'unique_image_paths': sorted(unique_paths)}
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'))
    print('Saved {}: triggered={}, unique semantic images={}'.format(
        args.output, triggered, len(unique_paths)), flush=True)


if __name__ == '__main__':
    main()
