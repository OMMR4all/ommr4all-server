from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from ..dataset import SymbolDetectionDataset, Dataset, SymbolDetectionDatasetTorch
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYMBOLS_PC_TORCH

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import PCTorchPredictor
        return PCTorchPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import PCTorchTrainer
        return PCTorchTrainer

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import SymbolsExperimenter
        return SymbolsExperimenter

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return SymbolDetectionDatasetTorch


Step.register(Meta)
