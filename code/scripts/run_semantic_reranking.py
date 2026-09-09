import argparse
import csv
import json
import os.path as osp
import numpy as np

from torchreid.semantic import (AttributeWeight, RankedCandidate,
                                SemanticArbitrationConfig, SymbolRegistry,
                                arbitrate_relative_ambiguity)

ATTRIBUTE_SUBSETS = {
    'A': ('upper_colour', 'lower_colour'),
    'B': ('upper_colour', 'lower_colour', 'upper_type'),
    'C': ('upper_colour', 'lower_colour', 'lower_type'),
    'D': ('upper_colour', 'lower_colour', 'upper_type', 'lower_type'),
    'E': ('upper_colour', 'lower_colour', 'upper_type', 'lower_type', 'bag')
}


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def filtered_attributes(item, enabled, threshold, registry):
    source, result = (item or {}).get('attributes', {}), {}
    for name in ATTRIBUTE_SUBSETS['E']:
        raw = source.get(name, {})
        value = raw.get('value') if isinstance(raw, dict) else raw
        confidence = raw.get('confidence', 0.0) if isinstance(raw, dict) else 1.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        result[name] = registry.normalize(name, value) if name in enabled and confidence >= threshold else 'unknown'
    return result


def new_stats():
    return {'aps': [], 'cmc': {1: [], 5: [], 10: []}}


def add_metrics(stats, order, query_pid, gallery_pids):
    matches = gallery_pids[order] == query_pid
    if not np.any(matches):
        return
    curve = np.minimum(1, matches.cumsum())
    for rank in stats['cmc']:
        stats['cmc'][rank].append(float(curve[rank - 1]))
    precision = matches.cumsum() / (np.arange(len(matches)) + 1)
    stats['aps'].append(float((precision * matches).sum() / matches.sum()))


def summarise(stats):
    return {'mAP': float(np.mean(stats['aps'])),
            'Rank-1': float(np.mean(stats['cmc'][1])),
            'Rank-5': float(np.mean(stats['cmc'][5])),
            'Rank-10': float(np.mean(stats['cmc'][10]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', required=True); p.add_argument('--candidates', required=True)
    p.add_argument('--attribute-cache', required=True); p.add_argument('--output', required=True)
    p.add_argument('--transitions-csv', default=None)
    p.add_argument('--lambdas', type=float, nargs='+', default=[.01, .02, .05])
    p.add_argument('--confidence-thresholds', type=float, nargs='+', default=[.25, .35, .45, .55])
    p.add_argument('--attribute-subsets', nargs='+', default=['A', 'B', 'C'])
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if osp.exists(args.output) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite'.format(args.output))
    if any(name not in ATTRIBUTE_SUBSETS for name in args.attribute_subsets):
        raise ValueError('attribute subsets must be A, B, C, D, or E')

    with np.load(args.artifact, allow_pickle=False) as a:
        qf, gf = normalize(a['query_features']), normalize(a['gallery_features'])
        qp, gp, qc, gc = a['query_pids'], a['gallery_pids'], a['query_camids'], a['gallery_camids']
        qpaths, gpaths = a['query_impaths'].astype(str), a['gallery_impaths'].astype(str)
    with open(args.candidates, encoding='utf-8') as f: candidate_payload = json.load(f)
    with open(args.attribute_cache, encoding='utf-8') as f: cache = json.load(f)['images']
    if candidate_payload.get('topk') != 20 or candidate_payload.get('delta') != .02:
        raise ValueError('Candidates must use Top-K=20 and delta=.02')
    registry, baseline_stats, report, transition_rows = SymbolRegistry(), new_stats(), {'ablations': {}}, []
    ambiguity_sizes = []
    combinations = [(subset, threshold, lam) for subset in args.attribute_subsets
                    for threshold in args.confidence_thresholds for lam in args.lambdas]

    for combination_index, (subset, threshold, lam) in enumerate(combinations):
        enabled = ATTRIBUTE_SUBSETS[subset]
        config = SemanticArbitrationConfig(lambda_sem=lam, normalize_adjustment=True,
                                           weights={name: AttributeWeight(1., 1.) for name in enabled})
        stats, trigger_count, nonzero_pairs = new_stats(), 0, 0
        agreement = {'true': [0, 0], 'false': [0, 0]}
        transitions = dict.fromkeys(('baseline_wrong_to_semantic_correct', 'baseline_correct_to_semantic_wrong',
                                     'baseline_wrong_to_semantic_wrong', 'baseline_correct_to_semantic_correct'), 0)
        for qi in range(len(qf)):
            scores = np.clip(qf[qi].dot(gf.T), -1, 1)
            visual = np.argsort(-scores, kind='mergesort')
            valid_order = visual[~((gp[visual] == qp[qi]) & (gc[visual] == qc[qi]))]
            topk_order = valid_order[:20]
            prefix = topk_order[scores[topk_order] >= scores[topk_order[0]] - .02]
            if combination_index == 0:
                ambiguity_sizes.append(len(prefix))
            if combination_index == 0:
                add_metrics(baseline_stats, valid_order, qp[qi], gp)
            query_attrs = filtered_attributes(cache.get(qpaths[qi]), enabled, threshold, registry)
            ranking = [RankedCandidate(gpaths[index], float(scores[index])) for index in prefix]
            result, audit = arbitrate_relative_ambiguity(
                query_attrs, ranking,
                lambda path: filtered_attributes(cache.get(path), enabled, threshold, registry),
                config, delta=.02, top_k=20)
            for gi in prefix:
                gallery_attrs = filtered_attributes(cache.get(gpaths[gi]), enabled, threshold, registry)
                pair_type = 'true' if gp[gi] == qp[qi] else 'false'
                for name in enabled:
                    if query_attrs[name] != 'unknown' and gallery_attrs[name] != 'unknown':
                        agreement[pair_type][1] += 1
                        agreement[pair_type][0] += int(query_attrs[name] == gallery_attrs[name])
            reranked_prefix = np.asarray([next(index for index in prefix if gpaths[index] == row['key']) for row in result])
            semantic_order = np.concatenate((reranked_prefix, valid_order[len(prefix):]))
            add_metrics(stats, semantic_order, qp[qi], gp)
            trigger_count += int(audit['invoked'])
            nonzero_pairs += sum(row['semantic_adjustment'] != 0 for row in audit['decisions'])
            baseline_correct = gp[valid_order[0]] == qp[qi]
            semantic_correct = gp[semantic_order[0]] == qp[qi]
            category = ('baseline_correct_to_semantic_correct' if baseline_correct and semantic_correct else
                        'baseline_correct_to_semantic_wrong' if baseline_correct else
                        'baseline_wrong_to_semantic_correct' if semantic_correct else
                        'baseline_wrong_to_semantic_wrong')
            transitions[category] += 1
            if category in ('baseline_wrong_to_semantic_correct', 'baseline_correct_to_semantic_wrong'):
                transition_rows.append({'attribute_subset': subset, 'confidence_threshold': threshold, 'lambda_sem': lam,
                                        'transition': category, 'query_index': qi, 'query_path': qpaths[qi],
                                        'query_pid': int(qp[qi]), 'baseline_top1_path': gpaths[valid_order[0]],
                                        'baseline_top1_pid': int(gp[valid_order[0]]), 'semantic_top1_path': gpaths[semantic_order[0]],
                                        'semantic_top1_pid': int(gp[semantic_order[0]])})
            if (qi + 1) % 500 == 0: print('{}: {}/{}'.format(subset, qi + 1, len(qf)))
        baseline = summarise(baseline_stats); value = summarise(stats)
        key = 'subset={}|confidence={:.2f}|lambda={:.3f}'.format(subset, threshold, lam)
        report['ablations'][key] = {'attribute_subset': subset, 'enabled_attributes': list(enabled),
            'confidence_threshold': threshold, 'lambda_sem': lam, 'metrics': value,
            'change_from_baseline': {name: value[name] - baseline[name] for name in value},
            'rank1_transitions': transitions,
            'net_rank1_correction_count': transitions['baseline_wrong_to_semantic_correct'] - transitions['baseline_correct_to_semantic_wrong'],
            'semantic_trigger_count': trigger_count, 'semantic_trigger_rate': trigger_count / len(qf),
            'semantic_nonzero_pair_count': nonzero_pairs,
            'attribute_agreement_rate': {kind: values[0] / values[1] if values[1] else None
                                         for kind, values in agreement.items()}}
        print('Completed {}'.format(key))
    report['baseline'] = summarise(baseline_stats)
    report['candidate_set'] = {'topk': 20, 'delta': .02,
                               'average_size': float(np.mean(ambiguity_sizes))}
    with open(args.output, 'w', encoding='utf-8') as f: json.dump(report, f, indent=2)
    columns = ['attribute_subset', 'confidence_threshold', 'lambda_sem', 'transition', 'query_index', 'query_path', 'query_pid', 'baseline_top1_path', 'baseline_top1_pid', 'semantic_top1_path', 'semantic_top1_pid']
    with open(args.transitions_csv or osp.splitext(args.output)[0] + '_rank1_transitions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(transition_rows)
    print('Saved {}'.format(args.output))


if __name__ == '__main__': main()
