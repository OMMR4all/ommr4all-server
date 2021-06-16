from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from ..dataset import SymbolDetectionDataset, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYMBOLS_PC

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import CenterNetPredictor
        return CenterNetPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import CenterNetTrainer
        return CenterNetTrainer

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import SymbolsExperimenter
        return SymbolsExperimenter

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return SymbolDetectionDataset


Step.register(Meta)
