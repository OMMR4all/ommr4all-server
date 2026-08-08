"""Cache of loaded predictors, living inside a task worker process.

The task workers (restapi/operationworker/taskworkermain.py) are long-lived,
so predictors -- and the model weights they hold -- can be reused between
tasks: repeated predictions skip both the torch import and the model loading.

The cache key contains the effective model path and its modification time:
selecting a different model in the client or retraining a model automatically
leads to a fresh predictor. The prediction params are part of the key as well,
since predictors capture them at construction time.
"""
import logging
import os
import pickle
from collections import OrderedDict
from typing import Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from database.model import Model
    from .algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmPredictorSettings

logger = logging.getLogger(__name__)

# each cached predictor keeps its model weights in memory (and VRAM), so hold
# only the most recently used ones
MAX_CACHED_PREDICTORS = 3

_cache: 'OrderedDict[tuple, AlgorithmPredictor]' = OrderedDict()


def _effective_model(settings: 'AlgorithmPredictorSettings') -> Optional['Model']:
    if settings.params.modelId:
        # mirrors the override in AlgorithmPredictor.__init__
        from database.model import Model
        return Model(settings.params.modelId)
    return settings.model


def _model_mtime(model: 'Model') -> float:
    for path in (model.meta_path, model.path):
        try:
            return os.path.getmtime(path)
        except OSError:
            continue
    return -1


def get_or_create(meta: Type['AlgorithmMeta'], settings: 'AlgorithmPredictorSettings') -> 'AlgorithmPredictor':
    model = _effective_model(settings)
    if model is None:
        return meta.create_predictor(settings)

    # before the cache lookup: a prediction served by a cached predictor still uses the model
    model.mark_used()

    key = (meta.type(), model.path, _model_mtime(model), pickle.dumps(settings.params))
    predictor = _cache.get(key)
    if predictor is not None:
        _cache.move_to_end(key)
        logger.info('Reusing cached predictor for {} ({})'.format(meta.type().name, model.path))
        return predictor

    predictor = meta.create_predictor(settings)
    _cache[key] = predictor
    while len(_cache) > MAX_CACHED_PREDICTORS:
        _cache.popitem(last=False)
    return predictor
