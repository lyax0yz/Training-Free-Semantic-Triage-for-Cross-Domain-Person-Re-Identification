import argparse
import csv
import json
import os

import numpy as np
from scipy import sparse


RANKS = (1, 5, 10, 20, 50, 100)


def l2_normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def valid_order(order, query_pid, query_camid, gallery_pids, gallery_camids):
    invalid = ((gallery_pids[order] == query_pid) &
               (gallery_camids[order] == query_camid))
    return order[~invalid]


def update_metrics(state, order, query_pid, gallery_pids):
    matches = gallery_pids[order] == query_pid
    if not np.any(matches):
        return None
    first_rank = int(np.flatnonzero(matches)[0]) + 1
    cumulative = matches.cumsum()
    precision = cumulative / (np.arange(len(matches)) + 1)
    state['aps'].append(float((precision * matches).sum() / matches.sum()))
    for rank in RANKS:
        state['recall'][rank] += int(first_rank <= rank)
    state['valid_queries'] += 1
    return first_rank


def summarize_metrics(state):
    count = state['valid_queries']
    result = {'mAP': float(np.mean(state['aps']))}
    for rank in RANKS:
        result['Rank-{}'.format(rank)] = state['recall'][rank] / count
    result.update({'Recall@10': result['Rank-10'],
                   'Recall@20': result['Rank-20'],
                   'Recall@50': result['Rank-50'],
                   'Recall@100': result['Rank-100']})
    return result


def new_metrics():
    return {'aps': [], 'recall': {rank: 0 for rank in RANKS}, 'valid_queries': 0}


def baseline_analysis(qf, gf, q_pids, g_pids, q_camids, g_camids, q_paths):
    metrics = new_metrics()
    rows = []
    best_ranks = np.empty(len(qf), dtype=np.int32)
    top1_correct = np.empty(len(qf), dtype=np.bool_)
    for qi in range(len(qf)):
        similarity = np.clip(qf[qi].dot(gf.T), -1., 1.)
        order = np.argsort(-similarity, kind='mergesort')
        order = valid_order(order, q_pids[qi], q_camids[qi], g_pids, g_camids)
        first_rank = update_metrics(metrics, order, q_pids[qi], g_pids)
        if first_rank is None:
            raise RuntimeError('Query {} has no valid gallery true match'.format(qi))
        best_ranks[qi] = first_rank
        top1_correct[qi] = first_rank == 1
        rows.append({'query_index': qi, 'query_path': q_paths[qi],
                     'query_pid': int(q_pids[qi]),
                     'baseline_best_true_rank': first_rank,
                     'baseline_rank1_correct': bool(first_rank == 1)})
        if (qi + 1) % 500 == 0:
            print('Baseline ranking {}/{}'.format(qi + 1, len(qf)), flush=True)
    return metrics, rows, best_ranks, top1_correct


def write_part_a(path_json, path_csv, metrics, baseline_rows, best_ranks):
    failures = best_ranks > 1
    unavailable20 = best_ranks > 20
    if int(failures.sum()) != 1491 or int(unavailable20.sum()) != 871:
        raise RuntimeError('Expected 1491 failures and 871 outside Top-20; got {} and {}'.format(
            int(failures.sum()), int(unavailable20.sum())))
    ranks_871 = best_ranks[unavailable20]
    bucket_counts = {
        'rank_21_50': int(((ranks_871 >= 21) & (ranks_871 <= 50)).sum()),
        'rank_51_100': int(((ranks_871 >= 51) & (ranks_871 <= 100)).sum()),
        'rank_gt_100': int((ranks_871 > 100).sum())}
    total = len(ranks_871)
    baseline = summarize_metrics(metrics)
    report = {
        'population': {'baseline_rank1_failures': int(failures.sum()),
                       'baseline_failures_without_true_match_top20': total},
        'best_true_rank_for_original_871': {
            key: {'count': value, 'percentage': value / total}
            for key, value in bucket_counts.items()},
        'cumulative_retrieval_recall_all_queries': {
            'Recall@10': baseline['Rank-10'], 'Recall@20': baseline['Rank-20'],
            'Recall@50': baseline['Rank-50'], 'Recall@100': baseline['Rank-100']},
        'original_871_recoverability': {
            'recoverable_by_expanding_top20_to_top50': bucket_counts['rank_21_50'],
            'additionally_recoverable_top50_to_top100': bucket_counts['rank_51_100'],
            'still_outside_top100': bucket_counts['rank_gt_100']}}
    with open(path_json, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    detailed = []
    for row in baseline_rows:
        rank = row['baseline_best_true_rank']
        if rank <= 20:
            continue
        bucket = 'rank_21_50' if rank <= 50 else 'rank_51_100' if rank <= 100 else 'rank_gt_100'
        detailed.append(dict(row, top100_bucket=bucket,
                             recoverable_top50=bool(rank <= 50),
                             recoverable_top100=bool(rank <= 100)))
    fields = list(detailed[0])
    with open(path_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(detailed)
    return report


def exact_initial_rank(features, width, block_size):
    """Return exact-distance Top-width neighbors and per-row max D^2."""
    count = len(features)
    ranks = np.empty((count, width), dtype=np.int32)
    row_max_squared = np.empty(count, dtype=np.float32)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarity = np.clip(features[start:stop].dot(features.T), -1., 1.)
        distance = 1. - similarity
        row_max_squared[start:stop] = np.max(distance * distance, axis=1)
        partition = np.argpartition(distance, width - 1, axis=1)[:, :width]
        partition_distance = np.take_along_axis(distance, partition, axis=1)
        for local in range(stop - start):
            # Distance first, gallery index second gives deterministic ties.
            tie_order = np.lexsort((partition[local], partition_distance[local]))
            ranks[start + local] = partition[local, tie_order]
        print('All-pair neighbors {}/{}'.format(stop, count), flush=True)
    return ranks, np.maximum(row_max_squared, 1e-12)


def build_sparse_v(features, initial_rank, row_max_squared, k1):
    count = len(features)
    row_indices, row_values = [], []
    half_width = int(np.around(k1 / 2.)) + 1
    for index in range(count):
        forward = initial_rank[index, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.where(backward == index)[0]]
        expansion = reciprocal.copy()
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, :half_width]
            candidate_backward = initial_rank[candidate_forward, :half_width]
            candidate_reciprocal = candidate_forward[
                np.where(candidate_backward == candidate)[0]]
            if len(np.intersect1d(candidate_reciprocal, reciprocal)) > (2. / 3.) * len(candidate_reciprocal):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion).astype(np.int32)
        raw_distance = 1. - np.clip(features[index].dot(features[expansion].T), -1., 1.)
        normalized = (raw_distance * raw_distance) / row_max_squared[index]
        weights = np.exp(-normalized).astype(np.float32)
        weights /= weights.sum()
        row_indices.append(expansion)
        row_values.append(weights)
        if (index + 1) % 1000 == 0:
            print('k-reciprocal encoding {}/{}'.format(index + 1, count), flush=True)
    indptr = np.zeros(count + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(values) for values in row_values])
    indices = np.concatenate(row_indices).astype(np.int32)
    values = np.concatenate(row_values).astype(np.float32)
    return sparse.csr_matrix((values, indices, indptr), shape=(count, count))


def local_query_expansion(v_matrix, initial_rank, k2):
    count = v_matrix.shape[0]
    rows = np.repeat(np.arange(count, dtype=np.int32), k2)
    columns = initial_rank[:, :k2].reshape(-1)
    values = np.full(len(rows), 1. / k2, dtype=np.float32)
    averaging = sparse.csr_matrix((values, (rows, columns)), shape=(count, count))
    expanded = averaging.dot(v_matrix).tocsr()
    expanded.eliminate_zeros()
    return expanded


def reranked_analysis(qf, gf, combined, initial_rank, row_max_squared, v_matrix,
                      q_pids, g_pids, q_camids, g_camids, q_paths,
                      baseline_rows, baseline_ranks, lambda_value):
    metrics = new_metrics()
    v_csc = v_matrix.tocsc()
    query_count, combined_count = len(qf), len(combined)
    rows = []
    reranked_ranks = np.empty(query_count, dtype=np.int32)
    for qi in range(query_count):
        overlap = np.zeros(combined_count, dtype=np.float32)
        q_start, q_stop = v_matrix.indptr[qi], v_matrix.indptr[qi + 1]
        for column, q_value in zip(v_matrix.indices[q_start:q_stop], v_matrix.data[q_start:q_stop]):
            c_start, c_stop = v_csc.indptr[column], v_csc.indptr[column + 1]
            candidate_rows = v_csc.indices[c_start:c_stop]
            candidate_values = v_csc.data[c_start:c_stop]
            overlap[candidate_rows] += np.minimum(q_value, candidate_values)
        jaccard = 1. - overlap[query_count:] / (2. - overlap[query_count:])
        raw_distance = 1. - np.clip(qf[qi].dot(gf.T), -1., 1.)
        original = (raw_distance * raw_distance) / row_max_squared[qi]
        final_distance = (1. - lambda_value) * jaccard + lambda_value * original
        order = np.argsort(final_distance, kind='mergesort')
        order = valid_order(order, q_pids[qi], q_camids[qi], g_pids, g_camids)
        first_rank = update_metrics(metrics, order, q_pids[qi], g_pids)
        if first_rank is None:
            raise RuntimeError('Query {} has no valid gallery true match after reranking'.format(qi))
        reranked_ranks[qi] = first_rank
        before = int(baseline_ranks[qi])
        movement = ('outside20_to_inside20' if before > 20 and first_rank <= 20 else
                    'inside20_to_outside20' if before <= 20 and first_rank > 20 else
                    'stayed_inside20' if before <= 20 else 'stayed_outside20')
        rows.append(dict(baseline_rows[qi], reranked_best_true_rank=first_rank,
                         reranked_rank1_correct=bool(first_rank == 1),
                         top20_movement=movement))
        if (qi + 1) % 250 == 0:
            print('Reranked evaluation {}/{}'.format(qi + 1, query_count), flush=True)
    return metrics, rows, reranked_ranks


def write_part_b(path_json, path_csv, baseline_metrics, reranked_metrics,
                 rows, baseline_ranks, reranked_ranks, k1, k2, lambda_value):
    original_unavailable = baseline_ranks > 20
    reranked_unavailable = reranked_ranks > 20
    outside_to_inside = (original_unavailable & ~reranked_unavailable)
    inside_to_outside = (~original_unavailable & reranked_unavailable)
    reranked_failures = reranked_ranks > 1
    baseline_summary, reranked_summary = summarize_metrics(baseline_metrics), summarize_metrics(reranked_metrics)
    report = {
        'method': 'Zhong et al. standard k-reciprocal reranking',
        'parameters': {'k1': k1, 'k2': k2, 'lambda_value': lambda_value,
                       'distance': 'cosine distance on L2-normalized OSNet features'},
        'baseline_metrics': baseline_summary,
        'reranked_metrics': reranked_summary,
        'reranked_failure_funnel': {
            'total_rank1_failures': int(reranked_failures.sum()),
            'true_match_absent_top20': int((reranked_failures & reranked_unavailable).sum()),
            'true_match_present_top20': int((reranked_failures & ~reranked_unavailable).sum()),
            'original_871_moved_into_top20': int(outside_to_inside.sum())},
        'true_match_top20_movement_all_queries': {
            'outside_top20_to_inside_top20': int(outside_to_inside.sum()),
            'inside_top20_to_outside_top20': int(inside_to_outside.sum()),
            'net_top20_pool_change': int(outside_to_inside.sum() - inside_to_outside.sum())}}
    with open(path_json, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    fields = list(rows[0])
    with open(path_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--part-a-json', required=True)
    parser.add_argument('--part-a-csv', required=True)
    parser.add_argument('--part-b-json', required=True)
    parser.add_argument('--part-b-csv', required=True)
    parser.add_argument('--k1', type=int, default=20)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda-value', type=float, default=.3)
    parser.add_argument('--distance-block-size', type=int, default=128)
    args = parser.parse_args()
    with np.load(args.artifact, allow_pickle=False) as data:
        qf, gf = l2_normalize(data['query_features']), l2_normalize(data['gallery_features'])
        q_pids, g_pids = data['query_pids'], data['gallery_pids']
        q_camids, g_camids = data['query_camids'], data['gallery_camids']
        q_paths = data['query_impaths'].astype(str)
    baseline_metrics, baseline_rows, baseline_ranks, _ = baseline_analysis(
        qf, gf, q_pids, g_pids, q_camids, g_camids, q_paths)
    write_part_a(args.part_a_json, args.part_a_csv, baseline_metrics,
                 baseline_rows, baseline_ranks)
    print('Saved Part A outputs', flush=True)

    combined = np.concatenate((qf, gf), axis=0)
    width = max(args.k1 + 1, args.k2, int(np.around(args.k1 / 2.)) + 1)
    initial_rank, row_max_squared = exact_initial_rank(
        combined, width, args.distance_block_size)
    v_matrix = build_sparse_v(combined, initial_rank, row_max_squared, args.k1)
    if args.k2 != 1:
        v_matrix = local_query_expansion(v_matrix, initial_rank, args.k2)
    reranked_metrics, movement_rows, reranked_ranks = reranked_analysis(
        qf, gf, combined, initial_rank, row_max_squared, v_matrix,
        q_pids, g_pids, q_camids, g_camids, q_paths,
        baseline_rows, baseline_ranks, args.lambda_value)
    write_part_b(args.part_b_json, args.part_b_csv, baseline_metrics, reranked_metrics,
                 movement_rows, baseline_ranks, reranked_ranks,
                 args.k1, args.k2, args.lambda_value)
    print('Saved Part B outputs', flush=True)


if __name__ == '__main__':
    main()
