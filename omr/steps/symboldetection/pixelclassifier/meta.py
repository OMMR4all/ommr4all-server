from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from ..dataset import SymbolDetectionDataset, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYMBOLS_PC

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import PCPredictor
        return PCPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import PCTrainer
        return PCTrainer

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return SymbolDetectionDataset

    @staticmethod
    def model_dir() -> str:
        return "pc_symbol_detection"


Step.register(Meta)
