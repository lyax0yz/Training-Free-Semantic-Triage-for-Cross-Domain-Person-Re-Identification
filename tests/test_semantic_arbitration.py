import unittest

from torchreid.semantic import (
    AttributeWeight, RankedCandidate, SemanticArbitrationConfig, SymbolRegistry,
    arbitrate_relative_ambiguity, compute_semantic_adjustment
)


class SemanticArbitrationTests(unittest.TestCase):
    def test_registry_normalizes_aliases_and_unknown(self):
        registry = SymbolRegistry()
        values = registry.normalize_attributes({'upper_colour': 'Grey', 'bag': 'back pack', 'hair_length': 'unobservable'})
        self.assertEqual(values['upper_colour'], 'gray')
        self.assertEqual(values['bag'], 'backpack')
        self.assertEqual(values['hair_length'], 'unknown')

    def test_registry_accepts_cached_attribute_payload(self):
        registry = SymbolRegistry()
        self.assertEqual(
            registry.normalize('upper_colour', {'value': 'blue', 'confidence': .72}),
            'blue'
        )

    def test_adjustment_matches_conflicts_and_abstains(self):
        config = SemanticArbitrationConfig(weights={
            'upper_colour': AttributeWeight(2.0, 3.0),
            'bag': AttributeWeight(1.0, 4.0)
        })
        score, log = compute_semantic_adjustment(
            {'upper_colour': 'blue', 'bag': 'backpack', 'hair_colour': 'unknown'},
            {'upper_colour': 'blue', 'bag': 'handbag', 'hair_colour': 'black'}, config
        )
        self.assertEqual(score, -2.0)  # +2 upper match, -4 bag conflict; all others abstain.
        decisions = {item['attribute']: item for item in log}
        self.assertEqual(decisions['upper_colour']['decision'], 'match_bonus')
        self.assertEqual(decisions['bag']['decision'], 'conflict_penalty')
        self.assertEqual(decisions['hair_colour']['decision'], 'abstain_unknown')

    def test_relative_arbitration_only_changes_ambiguity_prefix(self):
        config = SemanticArbitrationConfig(lambda_sem=0.05, weights={
            'upper_colour': AttributeWeight(1.0, 1.0)
        })
        ranking = [RankedCandidate('wrong', .90), RankedCandidate('match', .89), RankedCandidate('outside', .86)]
        attributes = {
            'wrong': {'upper_colour': 'red'}, 'match': {'upper_colour': 'blue'}, 'outside': {'upper_colour': 'blue'}
        }
        result, audit = arbitrate_relative_ambiguity(
            {'upper_colour': 'blue'}, ranking, attributes.get, config, delta=.02, top_k=20
        )
        self.assertTrue(audit['invoked'])
        self.assertEqual(audit['candidate_set_size'], 2)
        self.assertEqual([item['key'] for item in result], ['match', 'wrong', 'outside'])
        self.assertEqual(result[2]['final_score'], .86)
        self.assertEqual(result[2]['semantic_adjustment'], 0.0)


if __name__ == '__main__':
    unittest.main()
