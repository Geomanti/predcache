"""predcache: persistent prediction caching and batched inference for ML models."""

__version__ = "0.2.0"

from .cache import PredictionCache, CacheEntry
from .batched import BatchedInferenceRunner, InferenceFn
from .windowing import FeatureWindowAssembler
from .observability import otel_available

__all__ = [
    "PredictionCache",
    "CacheEntry",
    "BatchedInferenceRunner",
    "FeatureWindowAssembler",
    "otel_available",
]