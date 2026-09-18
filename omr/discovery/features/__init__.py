"""Registry of feature extractors."""
from typing import Callable, Dict

from omr.discovery.config import FeatureConfig
from omr.discovery.features.base import SpatialFeatureMap, SymbolFeatureExtractor, l2_normalize
from omr.discovery.features.dino import DinoFeatureExtractor
from omr.discovery.features.stub import StubFeatureExtractor

FEATURE_EXTRACTORS: Dict[str, Callable[[FeatureConfig], SymbolFeatureExtractor]] = {
    'dino': DinoFeatureExtractor,
    'stub': StubFeatureExtractor,
}


def build_feature_extractor(cfg: FeatureConfig) -> SymbolFeatureExtractor:
    try:
        factory = FEATURE_EXTRACTORS[cfg.backend]
    except KeyError:
        raise ValueError('unknown feature backend {!r} (known: {})'.format(
            cfg.backend, ', '.join(sorted(FEATURE_EXTRACTORS))))
    return factory(cfg)


__all__ = ['FEATURE_EXTRACTORS', 'build_feature_extractor', 'SpatialFeatureMap',
           'SymbolFeatureExtractor', 'l2_normalize', 'DinoFeatureExtractor',
           'StubFeatureExtractor']
