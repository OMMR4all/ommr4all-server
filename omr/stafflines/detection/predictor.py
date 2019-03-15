from abc import ABC, abstractmethod
from database.file_formats.pcgts import MusicLines, PcGts
from enum import Enum
from typing import NamedTuple, List, Generator, Optional
from omr.dataset.pcgtsdataset import RegionLineMaskData


class PredictorParameters(NamedTuple):
    checkpoints: Optional[List[str]]
    full_page: bool
    gray: bool
    pad: int = 0
    extract_region_only: bool = False
    target_line_space_height: int = 10


class StaffLinesModelType(Enum):
    PIXEL_CLASSIFIER = 0


class PredictionResult(NamedTuple):
    music_lines: MusicLines             # Music lines in global (page coords)
    music_lines_local: MusicLines       # Music lines in local (cropped line if not full page)
    line: RegionLineMaskData


PredictionType = Generator[PredictionResult, None, None]


class StaffLinesPredictor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        pass


def create_staff_line_predictor(detector_type: StaffLinesModelType, params: PredictorParameters) -> StaffLinesPredictor:
    if detector_type == StaffLinesModelType.PIXEL_CLASSIFIER:
        from .pixelclassifier.predictor import BasicStaffLinePredictor
        return BasicStaffLinePredictor(params)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))


