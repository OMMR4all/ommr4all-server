from abc import ABC, abstractmethod
from omr.stafflines.detection.predictor import StaffLinesModelType
from database import DatabaseBook


class StaffLinesTrainer(ABC):
    def __init__(self, target_book: DatabaseBook):
        super().__init__()
        self.book: DatabaseBook = target_book

    @abstractmethod
    def train(self):
        return None


def create_staff_line_trainer(detector_type: StaffLinesModelType, b: DatabaseBook) -> StaffLinesTrainer:
    if detector_type == StaffLinesModelType.PIXEL_CLASSIFIER:
        from .pixelclassifier.trainer import BasicStaffLinesTrainer
        return BasicStaffLinesTrainer(b)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))
