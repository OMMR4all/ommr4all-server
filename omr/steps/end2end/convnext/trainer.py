from typing import Type

from omr.steps.algorithm import AlgorithmMeta

from ..trainer import End2EndTrainerBase


class ConvNeXtTrainer(End2EndTrainerBase):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from .meta import Meta
        return Meta
