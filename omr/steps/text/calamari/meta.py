from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from ..dataset import TextDataset, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.OCR_CALAMARI

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import CalamariPredictor
        return CalamariPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import CalamariTrainer
        return CalamariTrainer

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return TextDataset


Step.register(Meta)
