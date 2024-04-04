from omr.dataset import Dataset
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import DropCapitalPredictor
        return DropCapitalPredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        raise NotImplementedError()

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        from omr.steps.layout.drop_capitals.dataset import DropCapitalDatasetDataset
        return DropCapitalDatasetDataset


Step.register(Meta)
