import argparse
import csv
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic'))
from scheme2 import (ATTRIBUTE_NAMES, PedestrianSymbolRegistry, Scheme2Config,
                     compute_scheme2_adjustment)  # noqa: E402


CONFIGURATIONS = {
    'A': {'label': 'OSNet baseline', 'semantic': False},
    'B': {'label': 'existing single-view SigLIP2 colour-only', 'legacy_colour': True},
    'C': {'label': 'proper attributes only', 'single_view': True,
          'uncertainty_gate': False, 'quality_mask': False},
    'D': {'label': 'proper attributes + SymbolRegistry + multi-view voting',
          'single_view': False, 'uncertainty_gate': False, 'quality_mask': False},
    'E': {'label': 'D + uncertainty gating', 'single_view': False,
          'uncertainty_gate': True, 'quality_mask': False},
    'F': {'label': 'E + quality masking', 'single_view': False,
          'uncertainty_gate': True, 'quality_mask': True},
}


LEGACY_COLOUR_VOCABULARIES = {
    'upper_colour': {'black', 'white', 'gray', 'red', 'blue', 'green',
                     'yellow', 'brown', 'pink', 'purple', 'orange',
                     'multicolour', 'other'},
    'lower_colour': {'black', 'white', 'gray', 'blue', 'brown', 'green',
                     'red', 'beige', 'multicolour', 'other'},
}


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def new_metrics():
    return {'aps': [], 'cmc': {1: [], 5: [], 10: []}}


def add_metrics(stats, order, query_pid, gallery_pids):
    matches = gallery_pids[order] == query_pid
    if not np.any(matches):
        return False
    cumulative = np.minimum(1, matches.cumsum())
    for rank in stats['cmc']:
        stats['cmc'][rank].append(float(cumulative[rank - 1]))
    precision = matches.cumsum() / (np.arange(len(matches), dtype=np.float64) + 1.)
    stats['aps'].append(float((precision * matches).sum() / matches.sum()))
    return True


def summarize_metrics(stats):
    return {'mAP': float(np.mean(stats['aps'])),
            'Rank-1': float(np.mean(stats['cmc'][1])),
            'Rank-5': float(np.mean(stats['cmc'][5])),
            'Rank-10': float(np.mean(stats['cmc'][10]))}


def legacy_colour_attributes(cache_item, registry, threshold=.25):
    source = (cache_item or {}).get('attributes', {})
    output = {}
    for name in ('upper_colour', 'lower_colour'):
        raw = source.get(name, {})
        confidence = float(raw.get('confidence', 0.)) if isinstance(raw, dict) else 1.
        value = raw.get('value') if isinstance(raw, dict) else raw
        token = str(value).strip().lower().replace(' ', '_').replace('-', '_')
        token = {'grey': 'gray', 'multicolor': 'multicolour',
                 'multi_color': 'multicolour'}.get(token, token)
        symbol = (token if confidence >= threshold and
                  token in LEGACY_COLOUR_VOCABULARIES[name] else 'unknown')
        output[name] = {'value': symbol, 'confidence': confidence,
                        'vote_agreement': 1. if symbol != 'unknown' else 0.}
    return output


def legacy_colour_adjustment(query, gallery):
    decisions = []
    total = 0.
    reliabilities = []
    for name in ('upper_colour', 'lower_colour'):
        q, g = query[name], gallery[name]
        known = q['value'] != 'unknown' and g['value'] != 'unknown'
        reliability = min(q['confidence'], g['confidence']) if known else 0.
        if not known:
            reason, contribution = 'unknown', 0.
        elif q['value'] == g['value']:
            reason, contribution = 'match', 1.
        else:
            reason, contribution = 'conflict', -1.
        total += contribution
        if known:
            reliabilities.append(reliability)
        decisions.append({'attribute': name, 'query_value': q['value'],
                          'gallery_value': g['value'], 'query_confidence': q['confidence'],
                          'gallery_confidence': g['confidence'], 'query_vote_agreement': q['vote_agreement'],
                          'gallery_vote_agreement': g['vote_agreement'], 'reliability': reliability,
                          'accepted': known, 'reason': reason, 'weight': 1.,
                          'contribution': contribution})
    adjustment = total / 2.
    return {'semantic_adjustment': adjustment,
            'effective_semantic_adjustment': adjustment,
            'accepted_attribute_count': sum(item['accepted'] for item in decisions),
            'abstained_attribute_count': sum(not item['accepted'] for item in decisions),
            'mean_accepted_reliability': (sum(reliabilities) / len(reliabilities)
                                          if reliabilities else 0.),
            'pair_quality': 1., 'quality_allowed': True, 'decisions': decisions}


def raw_original_attributes(cache_item):
    """Ablation C: proper schema, original view, no registry or voting."""
    source = (cache_item or {}).get('raw_views', {}).get('original', {})
    output = {}
    for name in ATTRIBUTE_NAMES:
        raw = source.get(name, {})
        value = str(raw.get('value', 'unknown')).strip().lower().replace(' ', '_')
        confidence = float(raw.get('confidence', 0.))
        output[name] = {'value': value, 'confidence': confidence,
                        'vote_agreement': 1. if value != 'unknown' else 0.,
                        'num_valid_views': int(value != 'unknown')}
    return output


def initialize_state(attribute_count):
    return {
        'metrics': new_metrics(), 'wrong_to_correct': 0, 'correct_to_wrong': 0,
        'score_modified_queries': 0, 'ranking_changed_queries': 0,
        'candidate_pair_count': 0, 'accepted_count': 0,
        'possible_attribute_count': 0, 'attribute_count': attribute_count,
        'agreement': {'true': {'matches': 0, 'accepted': 0},
                      'false': {'matches': 0, 'accepted': 0}},
        'attributes': {name: {'comparisons': 0, 'known': 0, 'accepted': 0,
                              'reliability_sum_known': 0.}
                       for name in ATTRIBUTE_NAMES},
        'successful_reliability': [], 'harmful_reliability': [],
        'cases': {'corrected': [], 'harmful': [], 'reachable_not_corrected': []},
    }


def attributes_for(configuration, path, legacy_cache, scheme_cache, registry):
    if CONFIGURATIONS[configuration].get('legacy_colour'):
        return legacy_colour_attributes(legacy_cache.get(path), registry)
    item = scheme_cache.get(path, {})
    if CONFIGURATIONS[configuration].get('single_view'):
        return raw_original_attributes(item)
    return item.get('aggregated_attributes', item.get('attributes', {}))


def pair_adjustment(configuration, query_attributes, gallery_attributes, config):
    if CONFIGURATIONS[configuration].get('legacy_colour'):
        return legacy_colour_adjustment(query_attributes, gallery_attributes)
    definition = CONFIGURATIONS[configuration]
    return compute_scheme2_adjustment(
        query_attributes, gallery_attributes, config,
        use_uncertainty_gate=definition['uncertainty_gate'],
        use_quality_mask=definition['quality_mask'])


def update_pair_statistics(state, detail, is_true):
    state['candidate_pair_count'] += 1
    state['accepted_count'] += detail['accepted_attribute_count']
    state['possible_attribute_count'] += state['attribute_count']
    pair_kind = 'true' if is_true else 'false'
    for decision in detail['decisions']:
        attribute = decision['attribute']
        values = state['attributes'][attribute]
        values['comparisons'] += 1
        known = decision['query_value'] != 'unknown' and decision['gallery_value'] != 'unknown'
        if known:
            values['known'] += 1
            values['reliability_sum_known'] += decision['reliability']
        if decision['accepted']:
            values['accepted'] += 1
            state['agreement'][pair_kind]['accepted'] += 1
            state['agreement'][pair_kind]['matches'] += int(decision['reason'] == 'match')


def serializable_attributes(attributes):
    return {name: dict(attributes.get(name, {})) for name in ATTRIBUTE_NAMES
            if name in attributes}


def image_attribute_statistics(paths, configuration, legacy_cache, scheme_cache, registry):
    names = ('upper_colour', 'lower_colour') if configuration == 'B' else ATTRIBUTE_NAMES
    values = {name: {'image_count': 0, 'known_count': 0,
                     'confidence_sum': 0., 'agreement_sum': 0.}
              for name in names}
    for path in paths:
        attributes = attributes_for(
            configuration, path, legacy_cache, scheme_cache, registry)
        for name in names:
            item = attributes.get(name, {})
            values[name]['image_count'] += 1
            if item.get('value', 'unknown') != 'unknown':
                values[name]['known_count'] += 1
                values[name]['confidence_sum'] += float(item.get('confidence', 0.))
                values[name]['agreement_sum'] += float(item.get('vote_agreement', 0.))
    return {name: {
        'known_rate': item['known_count'] / float(item['image_count']),
        'mean_confidence_when_known': (item['confidence_sum'] / item['known_count']
                                       if item['known_count'] else 0.),
        'mean_vote_agreement_when_known': (item['agreement_sum'] / item['known_count']
                                           if item['known_count'] else 0.)}
            for name, item in values.items()}


def build_case(case_type, query_index, query_pid, query_camid, query_path,
               true_index, false_index, scores, gallery_pids, gallery_camids,
               gallery_paths, query_attributes, gallery_attributes, detail_by_index,
               baseline_top_index, semantic_top_index):
    def candidate(index):
        detail = detail_by_index[index]
        return {
            'gallery_index': int(index), 'image_path': gallery_paths[index],
            'pid': int(gallery_pids[index]), 'camera_id': int(gallery_camids[index]),
            'visual_similarity': float(scores[index]),
            'attributes': serializable_attributes(gallery_attributes[index]),
            'attribute_decisions': detail['decisions'],
            'semantic_adjustment': detail['semantic_adjustment'],
            'effective_semantic_adjustment': detail['effective_semantic_adjustment'],
            'quality_scale': detail['pair_quality'],
            'quality_allowed': detail['quality_allowed'],
            'final_score': float(scores[index] +
                                 detail_by_index[index].get('lambda_sem', 0.) *
                                 detail['effective_semantic_adjustment'])}
    return {
        'case_type': case_type, 'query_index': int(query_index),
        'query': {'image_path': query_path, 'pid': int(query_pid),
                  'camera_id': int(query_camid),
                  'attributes': serializable_attributes(query_attributes)},
        'baseline_top1_gallery_index': int(baseline_top_index),
        'semantic_top1_gallery_index': int(semantic_top_index),
        'true_candidate': candidate(true_index),
        'competing_false_candidate': candidate(false_index),
        'true_minus_false_visual_gap': float(scores[true_index] - scores[false_index]),
    }


def flatten_case(case):
    true = case['true_candidate']; false = case['competing_false_candidate']
    return {
        'case_type': case['case_type'], 'query_index': case['query_index'],
        'query_path': case['query']['image_path'], 'query_pid': case['query']['pid'],
        'query_camera_id': case['query']['camera_id'],
        'true_path': true['image_path'], 'true_pid': true['pid'],
        'true_camera_id': true['camera_id'], 'true_visual_similarity': true['visual_similarity'],
        'true_semantic_adjustment': true['semantic_adjustment'],
        'true_effective_semantic_adjustment': true['effective_semantic_adjustment'],
        'true_quality_scale': true['quality_scale'], 'true_final_score': true['final_score'],
        'false_path': false['image_path'], 'false_pid': false['pid'],
        'false_camera_id': false['camera_id'], 'false_visual_similarity': false['visual_similarity'],
        'false_semantic_adjustment': false['semantic_adjustment'],
        'false_effective_semantic_adjustment': false['effective_semantic_adjustment'],
        'false_quality_scale': false['quality_scale'], 'false_final_score': false['final_score'],
        'true_minus_false_visual_gap': case['true_minus_false_visual_gap'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--legacy-siglip2-cache', required=True)
    parser.add_argument('--scheme2-cache', required=True)
    parser.add_argument('--config', default='configs/scheme2_market_to_duke.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-queries', type=int, default=0,
                        help='Debug only; zero evaluates the complete query set.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output_report = osp.join(args.output_dir, 'scheme2_ablation.json')
    if osp.exists(output_report) and not args.overwrite:
        raise FileExistsError('{} exists; pass --overwrite'.format(output_report))

    with open(args.config, encoding='utf-8') as handle:
        raw_config = json.load(handle)
    topk = int(raw_config.pop('candidate_topk'))
    delta = float(raw_config.pop('candidate_delta'))
    config = Scheme2Config.from_dict(raw_config)
    if topk != 20 or abs(delta - .02) > 1e-12:
        raise ValueError('Scheme 2 comparison is fixed at Top-K=20 and delta=0.02')
    with open(args.candidates, encoding='utf-8') as handle:
        candidate_cache = json.load(handle)
    if int(candidate_cache.get('topk')) != topk or abs(float(candidate_cache.get('delta')) - delta) > 1e-12:
        raise ValueError('Candidate cache protocol does not match the fixed Scheme 2 protocol')
    with open(args.legacy_siglip2_cache, encoding='utf-8') as handle:
        legacy_cache = json.load(handle)['images']
    with open(args.scheme2_cache, encoding='utf-8') as handle:
        scheme_payload = json.load(handle)
        scheme_cache = scheme_payload['images']

    with np.load(args.artifact, allow_pickle=False) as artifact:
        query_features = normalize(artifact['query_features'])
        gallery_features = normalize(artifact['gallery_features'])
        query_pids, gallery_pids = artifact['query_pids'], artifact['gallery_pids']
        query_camids, gallery_camids = artifact['query_camids'], artifact['gallery_camids']
        query_paths = artifact['query_impaths'].astype(str)
        gallery_paths = artifact['gallery_impaths'].astype(str)
    query_count = min(len(query_features), args.max_queries) if args.max_queries else len(query_features)
    required_paths = set(query_paths[:query_count])
    required_paths.update(candidate_cache['unique_image_paths'])
    missing_scheme = sorted(path for path in required_paths if path not in scheme_cache)
    if missing_scheme:
        raise ValueError('Scheme 2 cache is incomplete: {} required image(s) missing; first={}'.format(
            len(missing_scheme), missing_scheme[0]))

    registry = PedestrianSymbolRegistry()
    baseline_stats = new_metrics()
    states = {name: initialize_state(2 if name == 'B' else len(ATTRIBUTE_NAMES))
              for name in ('B', 'C', 'D', 'E', 'F')}
    ambiguity_sizes = []

    for query_index in range(query_count):
        scores = np.clip(query_features[query_index].dot(gallery_features.T), -1., 1.)
        visual_order = np.argsort(-scores, kind='mergesort')
        invalid = ((gallery_pids[visual_order] == query_pids[query_index]) &
                   (gallery_camids[visual_order] == query_camids[query_index]))
        valid = visual_order[~invalid]
        add_metrics(baseline_stats, valid, query_pids[query_index], gallery_pids)
        top = valid[:topk]
        ambiguity = top[scores[top] >= scores[top[0]] - delta]
        ambiguity_sizes.append(len(ambiguity))
        baseline_correct = gallery_pids[valid[0]] == query_pids[query_index]
        true_in_ambiguity = ambiguity[gallery_pids[ambiguity] == query_pids[query_index]]

        for name, state in states.items():
            if len(ambiguity) == 1:
                add_metrics(state['metrics'], valid, query_pids[query_index], gallery_pids)
                continue
            query_attributes = attributes_for(
                name, query_paths[query_index], legacy_cache, scheme_cache, registry)
            detail_by_index = {}
            gallery_attribute_by_index = {}
            rescored = []
            for original_rank, gallery_index in enumerate(ambiguity):
                gallery_attributes = attributes_for(
                    name, gallery_paths[gallery_index], legacy_cache, scheme_cache, registry)
                detail = pair_adjustment(name, query_attributes, gallery_attributes, config)
                detail['lambda_sem'] = config.lambda_sem
                final_score = scores[gallery_index] + config.lambda_sem * detail['effective_semantic_adjustment']
                detail_by_index[int(gallery_index)] = detail
                gallery_attribute_by_index[int(gallery_index)] = gallery_attributes
                rescored.append((-final_score, -scores[gallery_index], original_rank,
                                 int(gallery_index)))
                update_pair_statistics(
                    state, detail, gallery_pids[gallery_index] == query_pids[query_index])
            rescored.sort()
            semantic_prefix = np.asarray([row[3] for row in rescored], dtype=np.int64)
            semantic_order = np.concatenate((semantic_prefix, valid[len(ambiguity):]))
            add_metrics(state['metrics'], semantic_order, query_pids[query_index], gallery_pids)
            semantic_correct = gallery_pids[semantic_order[0]] == query_pids[query_index]
            ranking_changed = not np.array_equal(semantic_prefix, ambiguity)
            score_modified = any(abs(detail['effective_semantic_adjustment']) > 1e-12
                                 for detail in detail_by_index.values())
            state['ranking_changed_queries'] += int(ranking_changed)
            state['score_modified_queries'] += int(score_modified)

            if not baseline_correct and semantic_correct:
                state['wrong_to_correct'] += 1
                case_type = 'corrected'
            elif baseline_correct and not semantic_correct:
                state['correct_to_wrong'] += 1
                case_type = 'harmful'
            elif not baseline_correct and len(true_in_ambiguity) and not semantic_correct:
                case_type = 'reachable_not_corrected'
            else:
                case_type = None
            if case_type:
                true_index = (int(semantic_order[0]) if semantic_correct else
                              int(true_in_ambiguity[0]))
                false_index = (int(valid[0]) if case_type == 'corrected' else
                               int(semantic_order[0]))
                case = build_case(
                    case_type, query_index, query_pids[query_index],
                    query_camids[query_index], query_paths[query_index],
                    true_index, false_index, scores, gallery_pids, gallery_camids,
                    gallery_paths, query_attributes, gallery_attribute_by_index,
                    detail_by_index, int(valid[0]), int(semantic_order[0]))
                state['cases'][case_type].append(case)
                winner_detail = detail_by_index[int(semantic_order[0])]
                if case_type == 'corrected':
                    state['successful_reliability'].append(
                        winner_detail['mean_accepted_reliability'])
                elif case_type == 'harmful':
                    state['harmful_reliability'].append(
                        winner_detail['mean_accepted_reliability'])
        if (query_index + 1) % 100 == 0 or query_index + 1 == query_count:
            print('Processed {}/{} queries'.format(query_index + 1, query_count), flush=True)

    baseline = summarize_metrics(baseline_stats)
    results = {
        'protocol': {'source': 'market1501', 'target': 'dukemtmcreid',
                     'topk': topk, 'delta': delta, 'lambda_sem': config.lambda_sem,
                     'query_count': query_count,
                     'semantic_trigger_rule': 'ambiguity_size > 1',
                     'singleton_policy': 'preserve complete OSNet ranking; no semantic attribute access',
                     'ground_truth_policy': 'evaluation and post-hoc diagnostics only'},
        'semantic_model': scheme_payload.get('model'),
        'fixed_configuration': raw_config,
        'candidate_set': {'average_ambiguity_size': float(np.mean(ambiguity_sizes)),
                          'triggered_query_count': int(sum(size > 1 for size in ambiguity_sizes)),
                          'triggered_query_rate': float(np.mean(np.asarray(ambiguity_sizes) > 1))},
        'configurations': {'A': {'label': CONFIGURATIONS['A']['label'],
                                 'metrics': baseline}},
    }
    cache_paths = sorted(required_paths)
    results['image_attribute_statistics'] = {
        name: image_attribute_statistics(
            cache_paths, name, legacy_cache, scheme_cache, registry)
        for name in ('B', 'C', 'D', 'E', 'F')}
    ablation_rows = []
    for name, state in states.items():
        metrics = summarize_metrics(state['metrics'])
        attribute_report = {}
        for attribute, values in state['attributes'].items():
            if values['comparisons'] == 0:
                continue
            attribute_report[attribute] = {
                'known_pair_rate': values['known'] / float(values['comparisons']),
                'accepted_after_gating_rate': values['accepted'] / float(values['comparisons']),
                'mean_reliability_for_known_pairs': (
                    values['reliability_sum_known'] / values['known'] if values['known'] else 0.),
                'abstention_rate': 1. - values['accepted'] / float(values['comparisons'])}
        agreement = {
            pair_kind + '_match_rate': (values['matches'] / float(values['accepted'])
                                        if values['accepted'] else None)
            for pair_kind, values in state['agreement'].items()}
        configuration = {
            'label': CONFIGURATIONS[name]['label'], 'metrics': metrics,
            'change_from_baseline': {key: metrics[key] - baseline[key] for key in metrics},
            'wrong_to_correct_count': state['wrong_to_correct'],
            'correct_to_wrong_count': state['correct_to_wrong'],
            'net_rank1_correction_count': state['wrong_to_correct'] - state['correct_to_wrong'],
            'semantically_score_modified_query_count': state['score_modified_queries'],
            'ranking_changed_query_count': state['ranking_changed_queries'],
            'average_accepted_attributes_per_candidate_pair': (
                state['accepted_count'] / float(state['candidate_pair_count'])
                if state['candidate_pair_count'] else 0.),
            'semantic_abstention_rate': (
                1. - state['accepted_count'] / float(state['possible_attribute_count'])
                if state['possible_attribute_count'] else 1.),
            'harmful_case_count': state['correct_to_wrong'],
            'semantic_agreement': agreement,
            'average_semantic_reliability': {
                'successful_cases': (float(np.mean(state['successful_reliability']))
                                     if state['successful_reliability'] else None),
                'harmful_cases': (float(np.mean(state['harmful_reliability']))
                                  if state['harmful_reliability'] else None)},
            'per_attribute_uncertainty': attribute_report,
            'case_counts': {key: len(value) for key, value in state['cases'].items()},
        }
        results['configurations'][name] = configuration
        ablation_rows.append({
            'configuration': name, 'label': configuration['label'], **metrics,
            'mAP_change': configuration['change_from_baseline']['mAP'],
            'Rank-1_change': configuration['change_from_baseline']['Rank-1'],
            'wrong_to_correct': state['wrong_to_correct'],
            'correct_to_wrong': state['correct_to_wrong'],
            'net_rank1_correction': configuration['net_rank1_correction_count'],
            'semantically_modified_queries': state['score_modified_queries'],
            'ranking_changed_queries': state['ranking_changed_queries'],
            'average_accepted_attributes_per_pair': configuration['average_accepted_attributes_per_candidate_pair'],
            'semantic_abstention_rate': configuration['semantic_abstention_rate'],
            'harmful_case_count': configuration['harmful_case_count'],
            'true_match_semantic_agreement': agreement['true_match_rate'],
            'false_match_semantic_agreement': agreement['false_match_rate'],
            'successful_case_mean_reliability': configuration['average_semantic_reliability']['successful_cases'],
            'harmful_case_mean_reliability': configuration['average_semantic_reliability']['harmful_cases'],
        })

    best_name = max(('B', 'C', 'D', 'E', 'F'),
                    key=lambda key: (results['configurations'][key]['metrics']['Rank-1'],
                                     results['configurations'][key]['metrics']['mAP']))
    results['best_configuration_posthoc'] = best_name
    results['best_selection_note'] = ('Highest Rank-1 (mAP tie-break) among pre-declared fixed ablations; '
                                      'this is post-hoc reporting, not PID-based inference or tuning.')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_report, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    with open(osp.join(args.output_dir, 'scheme2_ablation.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader(); writer.writerows(ablation_rows)

    best_cases = states[best_name]['cases']
    diagnostic_dir = osp.join(args.output_dir, 'case_diagnostics')
    os.makedirs(diagnostic_dir, exist_ok=True)
    for case_type, cases in best_cases.items():
        with open(osp.join(diagnostic_dir, case_type + '.json'), 'w', encoding='utf-8') as handle:
            json.dump({'configuration': best_name, 'case_type': case_type,
                       'count': len(cases), 'cases': cases}, handle, indent=2)
        rows = [flatten_case(case) for case in cases]
        csv_path = osp.join(diagnostic_dir, case_type + '.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            else:
                handle.write('case_type,query_index\n')
    print('Best fixed configuration: {}'.format(best_name), flush=True)
    print('Saved {}'.format(output_report), flush=True)


if __name__ == '__main__':
    main()
