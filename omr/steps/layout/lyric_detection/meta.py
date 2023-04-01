from omr.dataset import Dataset
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.layout.lyric_detection.dataset import LyricsLocationDataset
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.LAYOUT_SIMPLE_LYRICS

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import TextLocationDetector
        return TextLocationDetector

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return LyricsLocationDataset


Step.register(Meta)
