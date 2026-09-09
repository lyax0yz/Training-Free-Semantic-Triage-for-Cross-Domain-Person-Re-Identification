from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ATTRIBUTE_NAMES = (
    'upper_colour', 'upper_type', 'lower_colour', 'lower_type', 'bag',
    'hair_colour', 'hair_length', 'accessory'
)
UNKNOWN = 'unknown'


class SymbolRegistry:
    """Normalize raw attribute strings to stable, closed-vocabulary tokens."""

    _VOCABULARY = {
        'upper_colour': ('black', 'white', 'gray', 'red', 'blue', 'green', 'yellow', 'brown', 'pink', 'purple', 'orange', 'multicolour', 'other'),
        'upper_type': ('tshirt', 'shirt', 'jacket', 'hoodie', 'coat', 'sweater', 'dress', 'other'),
        'lower_colour': ('black', 'white', 'gray', 'blue', 'brown', 'green', 'red', 'beige', 'multicolour', 'other'),
        'lower_type': ('trousers', 'jeans', 'shorts', 'skirt', 'dress', 'leggings', 'other'),
        'bag': ('none', 'backpack', 'handbag', 'shoulder_bag', 'crossbody_bag', 'other'),
        'hair_colour': ('black', 'brown', 'blonde', 'gray', 'red', 'other'),
        'hair_length': ('bald', 'short', 'medium', 'long', 'other'),
        'accessory': ('none', 'hat', 'glasses', 'scarf', 'mask', 'umbrella', 'other')
    }
    _ALIASES = {
        'grey': 'gray', 'dark_gray': 'gray', 'light_gray': 'gray', 'multi_color': 'multicolour', 'multicolor': 'multicolour',
        't_shirt': 'tshirt', 'tee': 'tshirt', 't-shirt': 'tshirt', 'long_sleeve_shirt': 'shirt',
        'coat_jacket': 'jacket', 'sweatshirt': 'sweater', 'no_bag': 'none', 'no_accessory': 'none',
        'back_pack': 'backpack', 'shoulderbag': 'shoulder_bag', 'cross_body_bag': 'crossbody_bag',
        'blond': 'blonde', 'no_hair': 'bald', 'unknown_value': 'unknown', 'n_a': 'unknown', 'na': 'unknown', 'none_known': 'unknown'
    }

    def normalize(self, attribute: str, value: Any) -> str:
        if attribute not in ATTRIBUTE_NAMES:
            raise KeyError('Unsupported pedestrian attribute: {}'.format(attribute))
        # Cached VLM/CLIP extractors commonly retain confidence alongside the
        # symbolic value. Ranking consumes the value only.
        if isinstance(value, Mapping):
            value = value.get('value', UNKNOWN)
        if value is None:
            return UNKNOWN
        token = str(value).strip().lower().replace(' ', '_').replace('-', '_')
        if not token or token in ('unknown', 'unk', 'n/a', 'null', 'none_of_the_above'):
            return UNKNOWN
        token = self._ALIASES.get(token, token)
        return token if token in self._VOCABULARY[attribute] else UNKNOWN

    def normalize_attributes(self, attributes: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        attributes = attributes or {}
        return {name: self.normalize(name, attributes.get(name, UNKNOWN)) for name in ATTRIBUTE_NAMES}


@dataclass(frozen=True)
class AttributeWeight:
    match_bonus: float = 1.0
    conflict_penalty: float = 1.0


@dataclass
class SemanticArbitrationConfig:
    """All semantic weights are explicit and independent of VLM extraction."""

    lambda_sem: float = 0.02
    normalize_adjustment: bool = False
    weights: Dict[str, AttributeWeight] = field(
        default_factory=lambda: {name: AttributeWeight() for name in ATTRIBUTE_NAMES}
    )

    def weight_for(self, attribute: str) -> AttributeWeight:
        if attribute not in ATTRIBUTE_NAMES:
            raise KeyError('Unsupported pedestrian attribute: {}'.format(attribute))
        return self.weights.get(attribute, AttributeWeight())


@dataclass(frozen=True)
class RankedCandidate:
    """A visual-ranking item. ``key`` is resolved by an external attribute provider."""

    key: Any
    visual_similarity: float


def compute_semantic_adjustment(
    query_attrs: Optional[Mapping[str, Any]],
    gallery_attrs: Optional[Mapping[str, Any]],
    config: Optional[SemanticArbitrationConfig] = None,
    registry: Optional[SymbolRegistry] = None
) -> Tuple[float, List[Dict[str, Any]]]:
    """Return raw semantic adjustment and per-attribute audit decisions.

    Unknown on either side abstains. Known equal tokens add a configured bonus;
    known unequal tokens apply a configured penalty.
    """
    config, registry = config or SemanticArbitrationConfig(), registry or SymbolRegistry()
    query = registry.normalize_attributes(query_attrs)
    gallery = registry.normalize_attributes(gallery_attrs)
    total = 0.0
    log: List[Dict[str, Any]] = []
    for attribute in ATTRIBUTE_NAMES:
        query_value, gallery_value = query[attribute], gallery[attribute]
        if UNKNOWN in (query_value, gallery_value):
            log.append({'attribute': attribute, 'query': query_value, 'gallery': gallery_value,
                        'decision': 'abstain_unknown', 'contribution': 0.0})
            continue
        weight = config.weight_for(attribute)
        if query_value == gallery_value:
            contribution, decision = weight.match_bonus, 'match_bonus'
        else:
            contribution, decision = -weight.conflict_penalty, 'conflict_penalty'
        total += contribution
        log.append({'attribute': attribute, 'query': query_value, 'gallery': gallery_value,
                    'decision': decision, 'contribution': contribution})
    if config.normalize_adjustment:
        enabled = tuple(config.weights) if config.weights else ATTRIBUTE_NAMES
        scale = sum(max(config.weight_for(name).match_bonus, config.weight_for(name).conflict_penalty)
                    for name in enabled)
        total = total / scale if scale else 0.0
    return total, log


def arbitrate_relative_ambiguity(
    query_attrs: Optional[Mapping[str, Any]],
    visual_ranking: Sequence[RankedCandidate],
    attribute_provider: Callable[[Any], Optional[Mapping[str, Any]]],
    config: Optional[SemanticArbitrationConfig] = None,
    registry: Optional[SymbolRegistry] = None,
    delta: float = 0.02,
    top_k: int = 20
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-rank only the retrieval-relative ambiguity prefix.

    ``visual_ranking`` must already be visual-score sorted and contain only
    standard valid ReID candidates. Candidates outside ``top_k`` or below
    ``top1 - delta`` retain their original visual score and rank.
    """
    if delta < 0 or top_k < 1:
        raise ValueError('delta must be non-negative and top_k must be positive')
    if not visual_ranking:
        return [], {'invoked': False, 'candidate_set_size': 0, 'decisions': []}
    config, registry = config or SemanticArbitrationConfig(), registry or SymbolRegistry()
    ranking = list(visual_ranking)
    if any(ranking[i].visual_similarity < ranking[i + 1].visual_similarity for i in range(len(ranking) - 1)):
        raise ValueError('visual_ranking must be sorted by descending visual_similarity')
    top1 = ranking[0].visual_similarity
    limit = min(top_k, len(ranking))
    ambiguity_size = 0
    while ambiguity_size < limit and ranking[ambiguity_size].visual_similarity >= top1 - delta:
        ambiguity_size += 1
    ambiguous = ranking[:ambiguity_size]
    decisions = []
    rescored = []
    for original_rank, candidate in enumerate(ambiguous):
        adjustment, decision_log = compute_semantic_adjustment(
            query_attrs, attribute_provider(candidate.key), config, registry
        )
        final_score = candidate.visual_similarity + config.lambda_sem * adjustment
        rescored.append({'key': candidate.key, 'visual_similarity': candidate.visual_similarity,
                         'semantic_adjustment': adjustment, 'final_score': final_score,
                         'original_rank': original_rank, 'decision_log': decision_log})
        decisions.append({'key': candidate.key, 'original_rank': original_rank,
                          'visual_similarity': candidate.visual_similarity,
                          'semantic_adjustment': adjustment, 'final_score': final_score,
                          'decision_log': decision_log})
    rescored.sort(key=lambda item: (-item['final_score'], -item['visual_similarity'], item['original_rank']))
    untouched = [
        {'key': candidate.key, 'visual_similarity': candidate.visual_similarity,
         'semantic_adjustment': 0.0, 'final_score': candidate.visual_similarity,
         'original_rank': index, 'decision_log': []}
        for index, candidate in enumerate(ranking[ambiguity_size:], start=ambiguity_size)
    ]
    return rescored + untouched, {
        'invoked': ambiguity_size > 1, 'candidate_set_size': ambiguity_size,
        'top1_visual_similarity': top1, 'delta': delta, 'top_k': top_k,
        'lambda_sem': config.lambda_sem, 'decisions': decisions
    }
