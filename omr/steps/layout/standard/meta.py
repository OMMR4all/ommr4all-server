from omr.dataset import Dataset
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.LAYOUT_COMPLEX_STANDARD

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import Predictor
        return Predictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None


Step.register(Meta)
