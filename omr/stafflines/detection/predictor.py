from abc import ABC, abstractmethod
from database.file_formats.pcgts import MusicLines
from enum import Enum
from database import DatabasePage


class StaffLinesModelType(Enum):
    PIXEL_CLASSIFIER = 0


class StaffLinesPredictor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def detect(self, binary_path: str, gray_path: str) -> MusicLines:
        return MusicLines()


def create_staff_line_predictor(detector_type: StaffLinesModelType, page: DatabasePage) -> StaffLinesPredictor:
    if detector_type == StaffLinesModelType.PIXEL_CLASSIFIER:
        from .pixelclassifier.predictor import BasicStaffLinePredictor
        return BasicStaffLinePredictor(page)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))


