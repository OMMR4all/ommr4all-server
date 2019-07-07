from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes
from omr.steps.step import Step
from ..dataset import PCDataset, Dataset


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.STAFF_LINES_PC

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import BasicStaffLinePredictor
        return BasicStaffLinePredictor

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import BasicStaffLinesTrainer
        return BasicStaffLinesTrainer

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return PCDataset

    @staticmethod
    def model_dir() -> str:
        return "pc_staff_lines"

    @staticmethod
    def best_model_name() -> str:
        return "model"


Step.register(Meta)
