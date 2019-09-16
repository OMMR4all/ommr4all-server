from typing import NamedTuple, List

from database import DatabasePage
from database.file_formats.pcgts import Page
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictionResult


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    line: RegionLineMaskData

    def pcgts(self):
        return self.line.operation.pcgts

    def page(self) -> Page:
        return self.line.operation.page

    def ds_page(self) -> DatabasePage:
        return self.page().location

    def to_dict(self):
        return {
            'page': self.ds_page().page,
            'book': self.ds_page().book.book,
        }

    def store_to_page(self):
        page = self.page()
        page.annotations.connections.clear()
        self.pcgts().to_file(self.ds_page().file('pcgts').local_path())
