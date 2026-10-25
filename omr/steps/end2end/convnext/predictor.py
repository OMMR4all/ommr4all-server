from typing import Type

from omr.steps.algorithm import AlgorithmMeta

from ..predictor import End2EndPredictor


class ConvNeXtPredictor(End2EndPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from .meta import Meta
        return Meta
