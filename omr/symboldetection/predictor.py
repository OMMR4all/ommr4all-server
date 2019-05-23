from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple
from database.file_formats.pcgts import *
from omr.symboldetection.dataset import SymbolDetectionDataset, RegionLineMaskData, SymbolDetectionDatasetParams
from enum import Enum


class SymbolDetectionPredictorParameters(NamedTuple):
    checkpoints: List[str]
    symbol_detection_params: SymbolDetectionDatasetParams = SymbolDetectionDatasetParams()


class PredictionResult(NamedTuple):
    symbols: List[Symbol]
    line: RegionLineMaskData


PredictionType = Generator[PredictionResult, None, None]


class SymbolDetectionPredictor(ABC):
    def __init__(self, params: SymbolDetectionPredictorParameters):
        self.params = params
        self.dataset: SymbolDetectionDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        self.dataset = SymbolDetectionDataset(pcgts_files, self.params.symbol_detection_params)
        return self._predict(self.dataset)

    @abstractmethod
    def _predict(self, dataset: SymbolDetectionDataset) -> PredictionType:
        pass


class PredictorTypes(Enum):
    PIXEL_CLASSIFIER = 0
    CALAMARI = 1
    PC_CALAMARI = 2

    def __str__(self):
        return self.name


def create_predictor(t: PredictorTypes, params: SymbolDetectionPredictorParameters) -> SymbolDetectionPredictor:
    if t == PredictorTypes.PIXEL_CLASSIFIER:
        from omr.symboldetection.pixelclassifier.predictor import PCPredictor
        return PCPredictor(params)
    elif t == PredictorTypes.CALAMARI:
        from omr.symboldetection.sequencetosequence.predictor import OMRPredictor
        return OMRPredictor(params)
    elif t == PredictorTypes.PC_CALAMARI:
        from omr.symboldetection.pcs2s.predictor import PCS2SPredictor
        return PCS2SPredictor(params)

    raise Exception('Invalid type {}'.format(type))
