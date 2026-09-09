import argparse
import json
import os
import os.path as osp
import numpy as np


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--delta', type=float, default=.02)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.topk < 1 or args.delta < 0:
        raise ValueError('topk must be positive and delta non-negative')
    output = osp.abspath(args.output)
    if osp.exists(output) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite'.format(output))
    os.makedirs(osp.dirname(output), exist_ok=True)
    with np.load(args.artifact, allow_pickle=False) as a:
        qf, gf = normalize_rows(a['query_features']), normalize_rows(a['gallery_features'])
        qp, gp, qc, gc = a['query_pids'], a['gallery_pids'], a['query_camids'], a['gallery_camids']
        qpaths, gpaths = a['query_impaths'].astype(str), a['gallery_impaths'].astype(str)
        metadata = json.loads(str(a['metadata_json'].item()))
    queries, unique_paths = [], set()
    for qi, scores in enumerate(np.clip(qf.dot(gf.T), -1, 1)):
        order = np.argsort(scores)[::-1]
        valid = ~((gp[order] == qp[qi]) & (gc[order] == qc[qi]))
        valid_order = order[valid]
        prefix = valid_order[:args.topk]
        size = int(np.count_nonzero(scores[prefix] >= scores[prefix[0]] - args.delta))
        candidates = prefix[:size]
        unique_paths.add(qpaths[qi])
        unique_paths.update(gpaths[candidates].tolist())
        queries.append({'query_index': qi, 'query_image_path': qpaths[qi],
                        'top1_visual_similarity': float(scores[prefix[0]]),
                        'candidate_gallery_indices': candidates.astype(int).tolist(),
                        'candidate_gallery_paths': gpaths[candidates].tolist(),
                        'candidate_visual_similarities': scores[candidates].astype(float).tolist()})
        if (qi + 1) % 100 == 0:
            print('Processed {}/{} queries'.format(qi + 1, len(qf)))
    report = {'artifact': osp.abspath(args.artifact), 'artifact_metadata': metadata,
              'topk': args.topk, 'delta': args.delta, 'queries': queries,
              'unique_image_paths': sorted(unique_paths)}
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Saved {} queries and {} unique images to {}'.format(len(queries), len(unique_paths), output))


if __name__ == '__main__':
    main()
