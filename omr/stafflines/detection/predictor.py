from abc import ABC, abstractmethod
from database.file_formats.pcgts import MusicLines, PcGts
from enum import Enum
from typing import NamedTuple, List, Generator, Optional
from omr.symboldetection.dataset import RegionLineMaskData
from omr.stafflines.detection.dataset import StaffLineDetectionDatasetParams
from linesegmentation.detection.settings import PostProcess


class StaffLinePredictorParameters(NamedTuple):
    checkpoints: Optional[List[str]]
    dataset_params: StaffLineDetectionDatasetParams = StaffLineDetectionDatasetParams()
    target_line_space_height: int = 10
    post_processing: PostProcess = PostProcess.BESTFIT
    best_fit_scale = 2.0

    num_staff_lines: int = 4
    min_num_staff_lines: int = 3


class StaffLinesModelType(Enum):
    PIXEL_CLASSIFIER = 0


class PredictionResult(NamedTuple):
    music_lines: MusicLines             # Music lines in global (page coords)
    music_lines_local: MusicLines       # Music lines in local (cropped line if not full page)
    line: RegionLineMaskData


class LineDetectionPredictorCallback(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def progress_updated(self, percentage: float):
        pass


PredictionType = Generator[PredictionResult, None, None]


class StaffLinesPredictor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, pcgts_files: List[PcGts], callback: Optional[LineDetectionPredictorCallback] = None) -> PredictionType:
        pass


def create_staff_line_predictor(detector_type: StaffLinesModelType, params: StaffLinePredictorParameters) -> StaffLinesPredictor:
    if detector_type == StaffLinesModelType.PIXEL_CLASSIFIER:
        from .pixelclassifier.predictor import BasicStaffLinePredictor
        return BasicStaffLinePredictor(params)

    else:
        raise Exception("Unknown staff line detector type: {}".format(detector_type))


