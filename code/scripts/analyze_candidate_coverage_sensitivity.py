import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic'))
from arbitration import (AttributeWeight, RankedCandidate,  # noqa: E402
                         SemanticArbitrationConfig, SymbolRegistry,
                         arbitrate_relative_ambiguity)

ATTRIBUTES = ('upper_colour', 'lower_colour')


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def cached_attributes(item, registry):
    source = (item or {}).get('attributes', {})
    result = {}
    for name in ATTRIBUTES:
        raw = source.get(name, {})
        value = raw.get('value') if isinstance(raw, dict) else raw
        confidence = raw.get('confidence', 0.0) if isinstance(raw, dict) else 1.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        result[name] = registry.normalize(name, value) if confidence >= .25 else 'unknown'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--attribute-cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with np.load(args.artifact, allow_pickle=False) as data:
        query_features = normalize(data['query_features'])
        gallery_features = normalize(data['gallery_features'])
        query_pids, gallery_pids = data['query_pids'], data['gallery_pids']
        query_camids, gallery_camids = data['query_camids'], data['gallery_camids']
        query_paths = data['query_impaths'].astype(str)
        gallery_paths = data['gallery_impaths'].astype(str)
    with open(args.attribute_cache, encoding='utf-8') as handle:
        cache = json.load(handle)['images']

    registry = SymbolRegistry()
    config = SemanticArbitrationConfig(
        lambda_sem=.05, normalize_adjustment=True,
        weights={name: AttributeWeight(1., 1.) for name in ATTRIBUTES})
    settings = ((20, .02), (20, .05), (50, .02), (50, .05))
    outcomes = {}

    for top_k, delta in settings:
        coverage = corrected = harmful = baseline_failures = 0
        for query_index in range(len(query_features)):
            scores = np.clip(query_features[query_index].dot(gallery_features.T), -1, 1)
            ranked = np.argsort(-scores, kind='mergesort')
            valid = ranked[~((gallery_pids[ranked] == query_pids[query_index]) &
                             (gallery_camids[ranked] == query_camids[query_index]))]
            baseline_correct = gallery_pids[valid[0]] == query_pids[query_index]
            if not baseline_correct:
                baseline_failures += 1
            top_candidates = valid[:top_k]
            ambiguity = top_candidates[scores[top_candidates] >= scores[top_candidates[0]] - delta]
            if not baseline_correct and np.any(gallery_pids[ambiguity] == query_pids[query_index]):
                coverage += 1

            query_attrs = cached_attributes(cache.get(query_paths[query_index]), registry)
            visual_prefix = [RankedCandidate(gallery_paths[index], float(scores[index]))
                             for index in ambiguity]
            result, _ = arbitrate_relative_ambiguity(
                query_attrs, visual_prefix,
                lambda path: cached_attributes(cache.get(path), registry),
                config, delta=delta, top_k=top_k)
            reranked_prefix = np.asarray([
                next(index for index in ambiguity if gallery_paths[index] == item['key'])
                for item in result])
            semantic_order = np.concatenate((reranked_prefix, valid[len(ambiguity):]))
            semantic_correct = gallery_pids[semantic_order[0]] == query_pids[query_index]
            corrected += int(not baseline_correct and semantic_correct)
            harmful += int(baseline_correct and not semantic_correct)

        if baseline_failures != 1491:
            raise RuntimeError('Expected 1491 baseline Rank-1 failures, found {}'.format(baseline_failures))
        key = 'topk={}|delta={:.2f}'.format(top_k, delta)
        outcomes[key] = {
            'topk': top_k, 'delta': delta,
            'baseline_rank1_failure_count': baseline_failures,
            'true_match_inside_ambiguity_count': coverage,
            'semantic_wrong_to_correct_count': corrected,
            'harmful_correct_to_wrong_count': harmful,
        }
        print('{}: coverage={}, corrected={}, harmful={}'.format(key, coverage, corrected, harmful))

    report = {
        'fixed_configuration': {
            'attributes': list(ATTRIBUTES), 'confidence_threshold': .25,
            'lambda_sem': .05,
            'description': 'Existing cached CLIP attributes and existing relative semantic reranking; only Top-K and delta vary.'
        },
        'results': outcomes,
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)


if __name__ == '__main__':
    main()
