from abc import ABC, abstractmethod
from database.file_formats.pcgts import Line
from enum import Enum
from typing import NamedTuple, List, Generator, Optional
from omr.dataset import DatasetParams, RegionLineMaskData


class PredictionResult(NamedTuple):
    music_lines: List[Line]             # Music lines in global (page coords)
    music_lines_local: List[Line]       # Music lines in local (cropped line if not full page)
    line: RegionLineMaskData


class LineDetectionPredictorCallback(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def progress_updated(self,
                         percentage: float,
                         n_pages: int = 0,
                         n_processed_pages: int = 0):
        pass


PredictionType = Generator[PredictionResult, None, None]


if __name__ == "__main__":
    from omr.steps.step import Step, AlgorithmTypes
    from database import DatabaseBook
    page = DatabaseBook('demo').pages()[0]
    params = StaffLinePredictorParameters(checkpoints=[page.book.local_default_models_path('pc_staff_lines/model')])
    pred = Step.create_predictor(AlgorithmTypes.STAFF_LINES_PC, params)
    pred = list(pred.predict([page.pcgts()]))[0]
    print(pred)
    print(len(pred.music_lines))

