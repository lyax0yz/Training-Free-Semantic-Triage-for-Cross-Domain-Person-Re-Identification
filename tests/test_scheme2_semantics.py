import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'torchreid' / 'semantic' / 'scheme2.py'
SPEC = importlib.util.spec_from_file_location('scheme2', str(MODULE_PATH))
scheme2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheme2)


def test_symbol_registry_normalizes_requested_synonyms():
    registry = scheme2.PedestrianSymbolRegistry()
    assert registry.normalize('upper_colour', 'navy') == 'blue'
    assert registry.normalize('lower_colour', 'dark gray') == 'gray'
    assert registry.normalize('upper_type', 'hooded sweatshirt') == 'hoodie'
    assert registry.normalize('lower_type', 'denim trousers') == 'jeans'
    assert registry.normalize('hat', 'maybe') == 'unknown'


def test_confidence_weighted_vote_and_agreement_abstention():
    predictions = [
        {'view': 'original', 'value': 'navy', 'confidence': .8},
        {'view': 'center', 'value': 'blue', 'confidence': .7},
        {'view': 'flip', 'value': 'black', 'confidence': .9},
    ]
    result = scheme2.aggregate_attribute_predictions(
        'upper_colour', predictions, minimum_agreement=.5)
    assert result['value'] == 'blue'
    assert abs(result['confidence'] - .5) < 1e-9
    assert abs(result['vote_agreement'] - 2. / 3.) < 1e-9
    strict = scheme2.aggregate_attribute_predictions(
        'upper_colour', predictions, minimum_agreement=.8)
    assert strict['value'] == 'unknown'


def _attributes(value='blue', confidence=.8, agreement=.75):
    return {name: {'value': value if 'colour' in name else 'unknown',
                   'confidence': confidence, 'vote_agreement': agreement,
                   'num_valid_views': 4}
            for name in scheme2.ATTRIBUTE_NAMES}


def test_unknown_abstains_and_reliability_gate_uses_confidence_and_vote():
    config = scheme2.Scheme2Config(reliability_threshold=.5)
    query = _attributes(confidence=.8, agreement=.75)
    gallery = _attributes(confidence=.8, agreement=.75)
    # reliability .6 passes for both colour attributes.
    result = scheme2.compute_scheme2_adjustment(
        query, gallery, config, use_uncertainty_gate=True,
        use_quality_mask=False)
    assert result['accepted_attribute_count'] == 2
    assert result['semantic_adjustment'] == 1.
    gallery['lower_colour']['vote_agreement'] = .5  # reliability .4
    result = scheme2.compute_scheme2_adjustment(
        query, gallery, config, use_uncertainty_gate=True,
        use_quality_mask=False)
    assert result['accepted_attribute_count'] == 1


def test_signed_adjustment_is_normalized_and_quality_scales_it():
    config = scheme2.Scheme2Config(reliability_threshold=0., minimum_pair_quality=0.)
    query = _attributes()
    gallery = _attributes()
    gallery['lower_colour']['value'] = 'black'
    raw = scheme2.compute_scheme2_adjustment(
        query, gallery, config, use_uncertainty_gate=False,
        use_quality_mask=False)
    assert raw['semantic_adjustment'] == 0.  # one colour match, one conflict
    gallery['lower_colour']['value'] = 'blue'
    scaled = scheme2.compute_scheme2_adjustment(
        query, gallery, config, use_uncertainty_gate=False,
        use_quality_mask=True)
    assert scaled['semantic_adjustment'] == 1.
    assert 0. < scaled['effective_semantic_adjustment'] < 1.


def test_low_pair_quality_falls_back_to_visual_score():
    config = scheme2.Scheme2Config(minimum_pair_quality=.9)
    query = _attributes(confidence=.2, agreement=.5)
    gallery = _attributes(confidence=.2, agreement=.5)
    result = scheme2.compute_scheme2_adjustment(
        query, gallery, config, use_uncertainty_gate=False,
        use_quality_mask=True)
    assert result['semantic_adjustment'] == 1.
    assert result['effective_semantic_adjustment'] == 0.
    assert not result['quality_allowed']
