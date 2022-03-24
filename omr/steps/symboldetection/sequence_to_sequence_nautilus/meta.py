from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from ..dataset import SymbolDetectionDataset, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYMBOLS_SEQUENCE_TO_SEQUENCE_NAUTILUS

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import OMRPredictor
        return OMRPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import OMRTrainer
        return OMRTrainer

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import SymbolsExperimenter
        return SymbolsExperimenter

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return SymbolDetectionDataset


Step.register(Meta)
