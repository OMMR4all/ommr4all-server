from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from ..dataset import TextDataset, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.OCR_NAUTILUS

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import PytorchPredictor
        return PytorchPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import PytorchTrainer
        return PytorchTrainer

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return TextDataset

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import TextExperimenter
        return TextExperimenter


Step.register(Meta)
