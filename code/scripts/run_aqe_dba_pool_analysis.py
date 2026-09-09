import argparse
import csv
import json

import numpy as np

from run_k_reciprocal_pool_analysis import (
    l2_normalize, new_metrics, summarize_metrics, update_metrics, valid_order
)


def topk_by_cosine(left, right, k, block_size, exclude_aligned_self=False):
    """Return deterministic cosine Top-K indices without using PID/camera labels."""
    result = np.empty((len(left), k), dtype=np.int32)
    for start in range(0, len(left), block_size):
        stop = min(start + block_size, len(left))
        similarity = np.clip(left[start:stop].dot(right.T), -1., 1.)
        if exclude_aligned_self:
            local_rows = np.arange(stop - start)
            similarity[local_rows, np.arange(start, stop)] = -np.inf
        partition = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        values = np.take_along_axis(similarity, partition, axis=1)
        for local in range(stop - start):
            ordering = np.lexsort((partition[local], -values[local]))
            result[start + local] = partition[local, ordering]
        print('Neighbor search {}/{}'.format(stop, len(left)), flush=True)
    return result


def uniform_expand(base, neighbor_features):
    """Uniformly average each base vector with its selected neighbors."""
    expanded = base + neighbor_features.sum(axis=1)
    return l2_normalize(expanded)


def apply_aqe(query_features, gallery_features, k, block_size):
    neighbors = topk_by_cosine(query_features, gallery_features, k, block_size)
    return uniform_expand(query_features, gallery_features[neighbors])


def apply_dba(gallery_features, k, block_size):
    neighbors = topk_by_cosine(
        gallery_features, gallery_features, k, block_size,
        exclude_aligned_self=True)
    return uniform_expand(gallery_features, gallery_features[neighbors])


def evaluate(name, qf, gf, q_pids, g_pids, q_camids, g_camids, q_paths,
             baseline_ranks=None):
    metrics = new_metrics()
    ranks = np.empty(len(qf), dtype=np.int32)
    rows = []
    for qi in range(len(qf)):
        similarity = np.clip(qf[qi].dot(gf.T), -1., 1.)
        order = np.argsort(-similarity, kind='mergesort')
        order = valid_order(order, q_pids[qi], q_camids[qi], g_pids, g_camids)
        first_rank = update_metrics(metrics, order, q_pids[qi], g_pids)
        if first_rank is None:
            raise RuntimeError('Query {} has no valid true match'.format(qi))
        ranks[qi] = first_rank
        row = {'query_index': qi, 'query_path': q_paths[qi],
               'query_pid': int(q_pids[qi]),
               '{}_best_true_rank'.format(name): first_rank,
               '{}_rank1_correct'.format(name): bool(first_rank == 1)}
        if baseline_ranks is not None:
            before = int(baseline_ranks[qi])
            movement = ('outside20_to_inside20' if before > 20 and first_rank <= 20 else
                        'inside20_to_outside20' if before <= 20 and first_rank > 20 else
                        'stayed_inside20' if before <= 20 else 'stayed_outside20')
            row['{}_top20_movement'.format(name)] = movement
        rows.append(row)
        if (qi + 1) % 500 == 0:
            print('{} evaluation {}/{}'.format(name, qi + 1, len(qf)), flush=True)
    return metrics, ranks, rows


def method_report(metrics, baseline_ranks, method_ranks):
    baseline_outside = baseline_ranks > 20
    method_outside = method_ranks > 20
    moved_in = int((baseline_outside & ~method_outside).sum())
    moved_out = int((~baseline_outside & method_outside).sum())
    return {
        'metrics': summarize_metrics(metrics),
        'top20_operable_pool_movement': {
            'original_unavailable_moved_into_top20': moved_in,
            'original_available_moved_out_of_top20': moved_out,
            'net_top20_operable_pool_change': moved_in - moved_out}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--k-reciprocal-report', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--aqe-k', type=int, default=10)
    parser.add_argument('--dba-k', type=int, default=10)
    parser.add_argument('--block-size', type=int, default=128)
    args = parser.parse_args()
    with np.load(args.artifact, allow_pickle=False) as data:
        qf, gf = l2_normalize(data['query_features']), l2_normalize(data['gallery_features'])
        q_pids, g_pids = data['query_pids'], data['gallery_pids']
        q_camids, g_camids = data['query_camids'], data['gallery_camids']
        q_paths = data['query_impaths'].astype(str)
    with open(args.k_reciprocal_report, encoding='utf-8') as handle:
        k_reciprocal = json.load(handle)
    k_reciprocal_metrics = dict(k_reciprocal['reranked_metrics'])
    k_reciprocal_metrics.setdefault('Recall@10', k_reciprocal_metrics['Rank-10'])

    baseline_metrics, baseline_ranks, baseline_rows = evaluate(
        'baseline', qf, gf, q_pids, g_pids, q_camids, g_camids, q_paths)
    if int((baseline_ranks > 1).sum()) != 1491 or int((baseline_ranks > 20).sum()) != 871:
        raise RuntimeError('Baseline does not reproduce 1491 failures / 871 outside Top-20')

    print('Applying AQE', flush=True)
    aqe_qf = apply_aqe(qf, gf, args.aqe_k, args.block_size)
    aqe_metrics, aqe_ranks, aqe_rows = evaluate(
        'aqe', aqe_qf, gf, q_pids, g_pids, q_camids, g_camids, q_paths, baseline_ranks)

    print('Applying DBA', flush=True)
    dba_gf = apply_dba(gf, args.dba_k, args.block_size)
    dba_metrics, dba_ranks, dba_rows = evaluate(
        'dba', qf, dba_gf, q_pids, g_pids, q_camids, g_camids, q_paths, baseline_ranks)

    print('Applying AQE after DBA', flush=True)
    combined_qf = apply_aqe(qf, dba_gf, args.aqe_k, args.block_size)
    combined_metrics, combined_ranks, combined_rows = evaluate(
        'aqe_dba', combined_qf, dba_gf, q_pids, g_pids, q_camids, g_camids,
        q_paths, baseline_ranks)

    baseline_summary = summarize_metrics(baseline_metrics)
    methods = {
        'AQE': method_report(aqe_metrics, baseline_ranks, aqe_ranks),
        'DBA': method_report(dba_metrics, baseline_ranks, dba_ranks),
        'AQE+DBA': method_report(combined_metrics, baseline_ranks, combined_ranks)}
    for method in methods.values():
        expected_net = int(round((method['metrics']['Recall@20'] - baseline_summary['Recall@20']) * len(qf)))
        actual_net = method['top20_operable_pool_movement']['net_top20_operable_pool_change']
        if expected_net != actual_net:
            raise RuntimeError('Top-20 movement accounting mismatch: {} != {}'.format(expected_net, actual_net))
    report = {
        'protocol': {
            'source_target': 'Market-1501 to DukeMTMC-reID',
            'features': 'L2-normalized existing OSNet features',
            'similarity': 'cosine',
            'pid_or_camera_used_for_augmentation': False,
            'iterations': 1,
            'weighting': 'uniform average',
            'aqe': {'gallery_neighbors': args.aqe_k, 'includes_query_self': True},
            'dba': {'gallery_neighbors_excluding_self': args.dba_k,
                    'includes_gallery_self': True},
            'combined_order': 'DBA first, then AQE against the augmented gallery'},
        'baseline': {'metrics': baseline_summary,
                     'top20_available_count': int((baseline_ranks <= 20).sum()),
                     'top20_unavailable_count': int((baseline_ranks > 20).sum())},
        'methods': methods,
        'existing_k_reciprocal': {
            'parameters': k_reciprocal['parameters'],
            'metrics': k_reciprocal_metrics,
            'top20_operable_pool_movement': {
                'original_unavailable_moved_into_top20': k_reciprocal['true_match_top20_movement_all_queries']['outside_top20_to_inside_top20'],
                'original_available_moved_out_of_top20': k_reciprocal['true_match_top20_movement_all_queries']['inside_top20_to_outside_top20'],
                'net_top20_operable_pool_change': k_reciprocal['true_match_top20_movement_all_queries']['net_top20_pool_change']}}}
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    detailed = []
    for qi in range(len(qf)):
        row = dict(baseline_rows[qi])
        for source in (aqe_rows, dba_rows, combined_rows):
            row.update({key: value for key, value in source[qi].items()
                        if key not in ('query_index', 'query_path', 'query_pid')})
        detailed.append(row)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    print('Saved AQE/DBA report and detailed CSV', flush=True)


if __name__ == '__main__':
    main()
