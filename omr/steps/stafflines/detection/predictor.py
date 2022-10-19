from abc import ABC, abstractmethod
from database.file_formats.pcgts import Line, BlockType, Block, Page
from typing import NamedTuple, List, Generator, Optional
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator
from database.database_page import DatabasePage


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    music_lines: List[Line]             # Music lines in global (page coords)
    music_lines_local: List[Line]       # Music lines in local (cropped line if not full page)
    line: RegionLineMaskData

    def pcgts(self):
        return self.line.operation.pcgts

    def page(self) -> Page:
        return self.line.operation.page

    def ds_page(self) -> DatabasePage:
        return self.page().location

    def to_dict(self):

        return {
            'staffs': [l.to_json() for l in self.music_lines],
            'page': self.ds_page().page,
            'book': self.ds_page().book.book,
        }

    def store_to_page(self):
        page = self.page()
        page.clear_blocks_of_type(BlockType.MUSIC)
        for ml in self.music_lines:
            page.blocks.append(
                Block(BlockType.MUSIC, lines=[ml])
            )

        self.pcgts().to_file(self.ds_page().file('pcgts').local_path())


class LineDetectionPredictorCallback(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def progress_updated(self,
                         percentage: float,
                         n_pages: int = 0,
                         n_processed_pages: int = 0):
        pass


class StaffLinePredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return len(page.pcgts().page.music_blocks()) == 0
