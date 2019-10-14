from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYMBOLS_SEQUENCE_MATCHER

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import Predictor
        return Predictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class():
        return None


Step.register(Meta)
