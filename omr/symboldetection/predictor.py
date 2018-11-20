from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple
from omr.datatypes import PcGts, Symbol
from omr.dataset.pcgtsdataset import MusicLineAndMarkedSymbol, PcGtsDataset
from enum import Enum


class PredictorParameters(NamedTuple):
    checkpoints: List[str]
    height: int = 80


class PredictionResult(NamedTuple):
    symbols: List[Symbol]
    line: MusicLineAndMarkedSymbol


PredictionType = Generator[PredictionResult, None, None]


class SymbolDetectionPredictor(ABC):
    def __init__(self, params: PredictorParameters):
        self.params = params
        self.dataset: PcGtsDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        self.dataset = PcGtsDataset(pcgts_files, gt_required=False, height=self.params.height)
        return self._predict(self.dataset)

    @abstractmethod
    def _predict(self, dataset: PcGtsDataset) -> PredictionType:
        pass


class PredictorTypes(Enum):
    PIXEL_CLASSIFIER = 0


def create_predictor(t: PredictorTypes, params: PredictorParameters) -> SymbolDetectionPredictor:
    if t == PredictorTypes.PIXEL_CLASSIFIER:
        from omr.symboldetection.pixelclassifier.predictor import PCPredictor
        return PCPredictor(params)

    raise Exception('Invalid type {}'.format(type))
