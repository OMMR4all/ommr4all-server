from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.DOCUMENT_ALIGNMENT

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import PreprocessingPredictor
        return PreprocessingPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None


Step.register(Meta)
