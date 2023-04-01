from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Dataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.POSTPROCESSING

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import PostprocessingPredictor
        return PostprocessingPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return None


Step.register(Meta)
