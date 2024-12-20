from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictor, AlgorithmTrainer, Type, AlgorithmTypes, Experimenter
from omr.steps.step import Step
from ..dataset import Dataset, PCDatasetTorch


class Meta(AlgorithmMeta):
    @staticmethod
    def type() -> AlgorithmTypes:
        return AlgorithmTypes.STAFF_LINES_PC_Torch

    @classmethod
    def predictor(cls) -> Type[AlgorithmPredictor]:
        from .predictor import BasicStaffLinePredictorTorch
        return BasicStaffLinePredictorTorch

    @classmethod
    def trainer(cls) -> Type[AlgorithmTrainer]:
        from .trainer import BasicStaffLinesTrainerTorch
        return BasicStaffLinesTrainerTorch

    @classmethod
    def experimenter(cls) -> Type[Experimenter]:
        from ..experimenter import StaffLinesExperimenter
        return StaffLinesExperimenter

    @staticmethod
    def dataset_class() -> Type[Dataset]:
        return PCDatasetTorch

    @staticmethod
    def best_model_name() -> str:
        return "model"


Step.register(Meta)
