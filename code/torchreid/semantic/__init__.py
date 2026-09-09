"""Pedestrian attribute normalization and retrieval-local semantic arbitration."""

from .arbitration import (
    ATTRIBUTE_NAMES, AttributeWeight, SemanticArbitrationConfig, SymbolRegistry,
    RankedCandidate, compute_semantic_adjustment, arbitrate_relative_ambiguity
)
from .scheme2 import (
    PedestrianSymbolRegistry, Scheme2Config, aggregate_attribute_predictions,
    aggregate_image_predictions, single_view_attributes,
    image_semantic_quality, compute_scheme2_adjustment
)

__all__ = [
    'ATTRIBUTE_NAMES', 'AttributeWeight', 'SemanticArbitrationConfig',
    'SymbolRegistry', 'RankedCandidate', 'compute_semantic_adjustment',
    'arbitrate_relative_ambiguity', 'PedestrianSymbolRegistry',
    'Scheme2Config', 'aggregate_attribute_predictions',
    'aggregate_image_predictions', 'single_view_attributes',
    'image_semantic_quality', 'compute_scheme2_adjustment'
]
