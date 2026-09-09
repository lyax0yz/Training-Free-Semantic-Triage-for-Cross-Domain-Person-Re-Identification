from __future__ import absolute_import

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


UNKNOWN = 'unknown'
ATTRIBUTE_NAMES = (
    'gender', 'build', 'age',
    'upper_colour', 'upper_type', 'lower_colour', 'lower_type',
    'backpack', 'handbag', 'hat'
)


class PedestrianSymbolRegistry(object):
    """Map extractor labels and common synonyms to stable discrete symbols."""

    VOCABULARIES = {
        'gender': ('male', 'female', UNKNOWN),
        'build': ('slim', 'average', 'stocky', UNKNOWN),
        'age': ('young_adult', 'middle_aged', 'older_adult', UNKNOWN),
        'upper_colour': ('black', 'white', 'gray', 'red', 'orange', 'yellow',
                         'green', 'blue', 'purple', 'pink', 'brown', UNKNOWN),
        'lower_colour': ('black', 'white', 'gray', 'red', 'orange', 'yellow',
                         'green', 'blue', 'purple', 'pink', 'brown', UNKNOWN),
        'upper_type': ('tshirt', 'shirt', 'jacket', 'coat', 'hoodie',
                       'sweater', 'dress', 'other', UNKNOWN),
        'lower_type': ('trousers', 'jeans', 'shorts', 'skirt', 'dress',
                       'leggings', 'other', UNKNOWN),
        'backpack': ('present', 'absent', UNKNOWN),
        'handbag': ('present', 'absent', UNKNOWN),
        'hat': ('present', 'absent', UNKNOWN),
    }

    COMMON_ALIASES = {
        '', 'none', 'n/a', 'na', 'not_visible', 'not visible', 'uncertain',
        'cannot_tell', 'cannot tell', 'indeterminate', 'unrecognizable'
    }

    ALIASES = {
        'gender': {
            'man': 'male', 'men': 'male', 'masculine': 'male',
            'woman': 'female', 'women': 'female', 'feminine': 'female'},
        'build': {
            'thin': 'slim', 'lean': 'slim', 'normal': 'average',
            'medium': 'average', 'heavy': 'stocky', 'large': 'stocky'},
        'age': {
            'young': 'young_adult', 'young adult': 'young_adult',
            'middle aged': 'middle_aged', 'middle-age': 'middle_aged',
            'middle age': 'middle_aged', 'old': 'older_adult',
            'older': 'older_adult', 'elderly': 'older_adult'},
        'upper_colour': {
            'grey': 'gray', 'charcoal': 'gray', 'dark_gray': 'gray',
            'dark_grey': 'gray', 'light_gray': 'gray', 'light_grey': 'gray',
            'navy': 'blue', 'navy_blue': 'blue', 'dark_blue': 'blue',
            'light_blue': 'blue', 'maroon': 'red', 'beige': 'brown',
            'tan': 'brown'},
        'lower_colour': {
            'grey': 'gray', 'charcoal': 'gray', 'dark_gray': 'gray',
            'dark_grey': 'gray', 'light_gray': 'gray', 'light_grey': 'gray',
            'navy': 'blue', 'navy_blue': 'blue', 'dark_blue': 'blue',
            'light_blue': 'blue', 'maroon': 'red', 'beige': 'brown',
            'tan': 'brown'},
        'upper_type': {
            't-shirt': 'tshirt', 't_shirt': 'tshirt', 'tee': 'tshirt',
            'hooded_sweatshirt': 'hoodie', 'hooded sweatshirt': 'hoodie',
            'pullover': 'sweater', 'blazer': 'jacket'},
        'lower_type': {
            'pants': 'trousers', 'long_pants': 'trousers',
            'denim_trousers': 'jeans', 'denim trousers': 'jeans',
            'denim_pants': 'jeans', 'short_pants': 'shorts',
            'tights': 'leggings'},
        'backpack': {
            'yes': 'present', 'backpack': 'present', 'with_backpack': 'present',
            'no': 'absent', 'no_backpack': 'absent', 'without_backpack': 'absent'},
        'handbag': {
            'yes': 'present', 'handbag': 'present', 'with_handbag': 'present',
            'no': 'absent', 'no_handbag': 'absent', 'without_handbag': 'absent'},
        'hat': {
            'yes': 'present', 'hat': 'present', 'with_hat': 'present',
            'no': 'absent', 'no_hat': 'absent', 'without_hat': 'absent'},
    }

    def normalize(self, attribute, value):
        if attribute not in self.VOCABULARIES or value is None:
            return UNKNOWN
        token = str(value).strip().lower().replace('-', '_')
        token = '_'.join(token.split())
        if token in {item.replace(' ', '_') for item in self.COMMON_ALIASES}:
            return UNKNOWN
        token = self.ALIASES.get(attribute, {}).get(token, token)
        return token if token in self.VOCABULARIES[attribute] else UNKNOWN


DEFAULT_ATTRIBUTE_WEIGHTS = {
    'gender': .75, 'build': .75, 'age': .75,
    'upper_colour': 1., 'lower_colour': 1.,
    'upper_type': .5, 'lower_type': .5,
    'backpack': .75, 'handbag': .75, 'hat': .75,
}


@dataclass
class Scheme2Config(object):
    attribute_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ATTRIBUTE_WEIGHTS))
    lambda_sem: float = .05
    min_vote_agreement: float = .5
    reliability_threshold: float = .25
    minimum_pair_quality: float = .25
    quality_confidence_weight: float = 1. / 3.
    quality_agreement_weight: float = 1. / 3.
    quality_known_weight: float = 1. / 3.

    @classmethod
    def from_dict(cls, values):
        config = cls()
        for key, value in values.items():
            if not hasattr(config, key):
                raise ValueError('Unknown Scheme2Config field: {}'.format(key))
            setattr(config, key, value)
        config.validate()
        return config

    def validate(self):
        if set(self.attribute_weights) != set(ATTRIBUTE_NAMES):
            raise ValueError('attribute_weights must contain exactly {}'.format(ATTRIBUTE_NAMES))
        if any(float(weight) < 0 for weight in self.attribute_weights.values()):
            raise ValueError('Attribute weights must be non-negative')
        for name in ('min_vote_agreement', 'reliability_threshold',
                     'minimum_pair_quality'):
            value = float(getattr(self, name))
            if not 0. <= value <= 1.:
                raise ValueError('{} must be in [0, 1]'.format(name))
        quality_sum = (self.quality_confidence_weight + self.quality_agreement_weight +
                       self.quality_known_weight)
        if quality_sum <= 0:
            raise ValueError('Quality component weights must have positive sum')


def _safe_probability(value):
    try:
        return max(0., min(1., float(value)))
    except (TypeError, ValueError):
        return 0.


def aggregate_attribute_predictions(attribute, predictions, registry=None,
                                    minimum_agreement=.5):
    registry = registry or PedestrianSymbolRegistry()
    valid = []
    normalized_views = []
    for prediction in predictions:
        raw_value = prediction.get('value', prediction.get('raw_value', UNKNOWN))
        confidence = _safe_probability(prediction.get('confidence', 0.))
        symbol = registry.normalize(attribute, raw_value)
        normalized = {'view': prediction.get('view'), 'raw_value': raw_value,
                      'value': symbol, 'confidence': confidence}
        normalized_views.append(normalized)
        if symbol != UNKNOWN:
            valid.append(normalized)
    if not valid:
        return {'value': UNKNOWN, 'confidence': 0., 'vote_agreement': 0.,
                'num_valid_views': 0, 'normalized_views': normalized_views}
    supports = {}
    counts = {}
    for item in valid:
        symbol = item['value']
        supports[symbol] = supports.get(symbol, 0.) + item['confidence']
        counts[symbol] = counts.get(symbol, 0) + 1
    winner = sorted(supports, key=lambda key: (-supports[key], -counts[key], key))[0]
    agreement = counts[winner] / float(len(valid))
    confidence = supports[winner] / float(len(valid))
    value = winner if agreement >= float(minimum_agreement) else UNKNOWN
    return {'value': value, 'confidence': confidence,
            'vote_agreement': agreement, 'num_valid_views': len(valid),
            'normalized_views': normalized_views}


def aggregate_image_predictions(raw_views, relevant_views, registry=None,
                                minimum_agreement=.5):
    registry = registry or PedestrianSymbolRegistry()
    aggregated = {}
    normalized = {}
    for attribute in ATTRIBUTE_NAMES:
        predictions = []
        for view_name in relevant_views[attribute]:
            raw = raw_views.get(view_name, {}).get(attribute)
            if raw is not None:
                item = dict(raw)
                item['view'] = view_name
                predictions.append(item)
        result = aggregate_attribute_predictions(
            attribute, predictions, registry, minimum_agreement)
        normalized[attribute] = result.pop('normalized_views')
        aggregated[attribute] = result
    return normalized, aggregated


def single_view_attributes(raw_views, registry=None, view_name='original'):
    """Return the proper-attribute/no-voting representation used in ablation C."""
    registry = registry or PedestrianSymbolRegistry()
    source = raw_views.get(view_name, {})
    output = {}
    for attribute in ATTRIBUTE_NAMES:
        raw = source.get(attribute, {})
        symbol = registry.normalize(attribute, raw.get('value', raw.get('raw_value')))
        confidence = _safe_probability(raw.get('confidence', 0.))
        output[attribute] = {
            'value': symbol, 'confidence': confidence,
            'vote_agreement': 1. if symbol != UNKNOWN else 0.,
            'num_valid_views': int(symbol != UNKNOWN)}
    return output


def image_semantic_quality(attributes, config=None, accepted_attributes=None):
    config = config or Scheme2Config()
    known_names = [name for name in ATTRIBUTE_NAMES
                   if attributes.get(name, {}).get('value', UNKNOWN) != UNKNOWN]
    known_rate = len(known_names) / float(len(ATTRIBUTE_NAMES))
    accepted_names = (known_names if accepted_attributes is None else
                      [name for name in accepted_attributes if name in known_names])
    accepted = [attributes.get(name, {}) for name in accepted_names]
    mean_confidence = (sum(_safe_probability(item.get('confidence', 0.)) for item in accepted) /
                       float(len(accepted))) if accepted else 0.
    mean_agreement = (sum(_safe_probability(item.get('vote_agreement', 0.)) for item in accepted) /
                      float(len(accepted))) if accepted else 0.
    denominator = (config.quality_confidence_weight + config.quality_agreement_weight +
                   config.quality_known_weight)
    quality = (config.quality_confidence_weight * mean_confidence +
               config.quality_agreement_weight * mean_agreement +
               config.quality_known_weight * known_rate) / denominator
    return {'quality': quality, 'mean_confidence': mean_confidence,
            'mean_vote_agreement': mean_agreement,
            'known_attribute_rate': known_rate, 'known_attribute_count': len(known_names),
            'accepted_attribute_count': len(accepted_names)}


def compute_scheme2_adjustment(query_attributes, gallery_attributes,
                               config=None, use_uncertainty_gate=True,
                               use_quality_mask=True):
    
    config = config or Scheme2Config()
    config.validate()
    decisions = []
    numerator = 0.
    denominator = 0.
    reliabilities = []
    for attribute in ATTRIBUTE_NAMES:
        query = query_attributes.get(attribute, {})
        gallery = gallery_attributes.get(attribute, {})
        q_value = query.get('value', UNKNOWN)
        g_value = gallery.get('value', UNKNOWN)
        q_confidence = _safe_probability(query.get('confidence', 0.))
        g_confidence = _safe_probability(gallery.get('confidence', 0.))
        q_agreement = _safe_probability(query.get('vote_agreement', 0.))
        g_agreement = _safe_probability(gallery.get('vote_agreement', 0.))
        mutual_confidence = min(q_confidence, g_confidence)
        reliability = mutual_confidence * min(q_agreement, g_agreement)
        known = q_value != UNKNOWN and g_value != UNKNOWN
        accepted = known and (not use_uncertainty_gate or
                              reliability >= config.reliability_threshold)
        if not known:
            reason = 'unknown'
        elif not accepted:
            reason = 'low_reliability'
        else:
            reason = 'match' if q_value == g_value else 'conflict'
        weight = float(config.attribute_weights[attribute])
        contribution = 0.
        if accepted and weight > 0:
            contribution = weight if q_value == g_value else -weight
            numerator += contribution
            denominator += weight
            reliabilities.append(reliability)
        decisions.append({
            'attribute': attribute, 'query_value': q_value,
            'gallery_value': g_value, 'query_confidence': q_confidence,
            'gallery_confidence': g_confidence,
            'query_vote_agreement': q_agreement,
            'gallery_vote_agreement': g_agreement,
            'mutual_confidence': mutual_confidence, 'reliability': reliability,
            'accepted': accepted, 'reason': reason, 'weight': weight,
            'contribution': contribution})
    adjustment = numerator / denominator if denominator else 0.
    accepted_names = [item['attribute'] for item in decisions if item['accepted']]
    query_quality = image_semantic_quality(query_attributes, config, accepted_names)
    gallery_quality = image_semantic_quality(gallery_attributes, config, accepted_names)
    pair_quality = min(query_quality['quality'], gallery_quality['quality'])
    quality_allowed = not use_quality_mask or pair_quality >= config.minimum_pair_quality
    effective = adjustment * pair_quality if use_quality_mask and quality_allowed else adjustment
    if use_quality_mask and not quality_allowed:
        effective = 0.
    return {
        'semantic_adjustment': adjustment,
        'effective_semantic_adjustment': effective,
        'accepted_attribute_count': sum(item['accepted'] for item in decisions),
        'abstained_attribute_count': sum(not item['accepted'] for item in decisions),
        'mean_accepted_reliability': (sum(reliabilities) / len(reliabilities)
                                      if reliabilities else 0.),
        'query_quality': query_quality, 'gallery_quality': gallery_quality,
        'pair_quality': pair_quality, 'quality_allowed': quality_allowed,
        'decisions': decisions}
