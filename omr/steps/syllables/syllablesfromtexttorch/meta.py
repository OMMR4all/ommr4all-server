from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Dataset, Experimenter
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import SyllablesFromTextPredictor
        return SyllablesFromTextPredictor

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import SyllablesExperimenter
        return SyllablesExperimenter

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None


Step.register(Meta)
