from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.LAYOUT_CONNECTED_COMPONENTS_SELECTION

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import Predictor
        return Predictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()


Step.register(Meta)
