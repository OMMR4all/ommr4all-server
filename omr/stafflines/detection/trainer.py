from abc import ABC, abstractmethod
from typing import Optional
from omr.stafflines.detection.predictor import StaffLinesModelType
from database import DatabaseBook
from omr.stafflines.detection.dataset import PCDatasetCallback


class StaffLinesDetectionTrainerCallback(PCDatasetCallback):
    def __init__(self):
        super().__init__()
        self.total_iters = 0
        self.early_stopping_iters = 0

    def init(self, total_iters, early_stopping_iters):
        self.total_iters = total_iters
        self.early_stopping_iters = early_stopping_iters

    def next_iteration(self, iter: int, loss: float, acc: float):
        pass

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        pass

    def early_stopping(self):
        pass

    def resolving_files(self):
        pass


class StaffLinesTrainer(ABC):
    def __init__(self, target_book: DatabaseBook):
        super().__init__()
        self.book: DatabaseBook = target_book

    @abstractmethod
    def train(self, callback: Optional[StaffLinesDetectionTrainerCallback]=None):
        return None


def create_staff_line_trainer(detector_type: StaffLinesModelType, b: DatabaseBook) -> StaffLinesTrainer:
    if detector_type == StaffLinesModelType.PIXEL_CLASSIFIER:
        from .pixelclassifier.trainer import BasicStaffLinesTrainer
        return BasicStaffLinesTrainer(b)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))
