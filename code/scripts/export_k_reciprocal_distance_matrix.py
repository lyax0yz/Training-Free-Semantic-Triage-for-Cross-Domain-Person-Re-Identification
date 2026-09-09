import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_k_reciprocal_pool_analysis import (  # noqa: E402
    RANKS, build_sparse_v, exact_initial_rank, l2_normalize,
    local_query_expansion, new_metrics, summarize_metrics,
    update_metrics, valid_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--k1', type=int, default=20)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=.3)
    parser.add_argument('--distance-block-size', type=int, default=128)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if osp.exists(args.output) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite'.format(args.output))
    with np.load(args.artifact, allow_pickle=False) as data:
        qf, gf = l2_normalize(data['query_features']), l2_normalize(data['gallery_features'])
        q_pids, g_pids = data['query_pids'], data['gallery_pids']
        q_camids, g_camids = data['query_camids'], data['gallery_camids']
    combined = np.concatenate((qf, gf), axis=0)
    width = max(args.k1 + 1, args.k2, int(np.around(args.k1 / 2.)) + 1)
    initial_rank, row_max_squared = exact_initial_rank(
        combined, width, args.distance_block_size)
    v_matrix = build_sparse_v(combined, initial_rank, row_max_squared, args.k1)
    if args.k2 != 1:
        v_matrix = local_query_expansion(v_matrix, initial_rank, args.k2)
    v_csc = v_matrix.tocsc()
    matrix = np.lib.format.open_memmap(
        args.output, mode='w+', dtype=np.float32, shape=(len(qf), len(gf)))
    metrics = new_metrics()
    combined_count = len(combined)
    for qi in range(len(qf)):
        overlap = np.zeros(combined_count, dtype=np.float32)
        q_start, q_stop = v_matrix.indptr[qi], v_matrix.indptr[qi + 1]
        for column, q_value in zip(v_matrix.indices[q_start:q_stop],
                                   v_matrix.data[q_start:q_stop]):
            c_start, c_stop = v_csc.indptr[column], v_csc.indptr[column + 1]
            rows, values = v_csc.indices[c_start:c_stop], v_csc.data[c_start:c_stop]
            overlap[rows] += np.minimum(q_value, values)
        jaccard = 1. - overlap[len(qf):] / (2. - overlap[len(qf):])
        raw = 1. - np.clip(qf[qi].dot(gf.T), -1., 1.)
        original = (raw * raw) / row_max_squared[qi]
        final = ((1. - args.lambda_value) * jaccard +
                 args.lambda_value * original).astype(np.float32)
        matrix[qi] = final
        order = valid_order(np.argsort(final, kind='mergesort'), q_pids[qi],
                            q_camids[qi], g_pids, g_camids)
        update_metrics(metrics, order, q_pids[qi], g_pids)
        if (qi + 1) % 250 == 0:
            matrix.flush()
            print('Exported {}/{} query rows'.format(qi + 1, len(qf)), flush=True)
    matrix.flush()
    report = {
        'method': 'Zhong et al. standard k-reciprocal reranking',
        'parameters': {'k1': args.k1, 'k2': args.k2,
                       'lambda_value': args.lambda_value},
        'distance_matrix': osp.abspath(args.output),
        'shape': list(matrix.shape), 'metrics': summarize_metrics(metrics)}
    with open(args.report, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
