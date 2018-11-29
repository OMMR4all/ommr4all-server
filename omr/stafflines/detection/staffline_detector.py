from abc import ABC, abstractmethod
from omr.datatypes import MusicLines
from enum import Enum
import main.book as book


class StaffLineDetectorType(Enum):
    DUMMY=0
    BASIC=1


class StaffLineDetector(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def detect(self, binary_path: str, gray_path: str) -> MusicLines:
        return MusicLines()


def create_staff_line_detector(detector_type: StaffLineDetectorType, page: book.Page) -> StaffLineDetector:
    if detector_type == StaffLineDetectorType.BASIC:
        from .basic_detector import BasicStaffLineDetector
        return BasicStaffLineDetector(page)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))