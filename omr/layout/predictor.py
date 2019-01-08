from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple
from omr.datatypes import PcGts, Symbol
from omr.dataset.pcgtsdataset import MusicLineAndMarkedSymbol, PcGtsDataset
from enum import Enum
import numpy as np


class PredictorParameters(NamedTuple):
    checkpoints: List[str]


class PredictionResult(NamedTuple):
    text_regions: List[np.ndarray]
    lyrics_regions: List[np.ndarray]
    music_regions: List[np.ndarray]
    drop_capital_regions: List[np.ndarray]


PredictionType = Generator[PredictionResult, None, None]


class LayoutAnalysisPredictor(ABC):
    def __init__(self, params: PredictorParameters):
        self.params = params
        self.dataset: PcGtsDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        return None


class PredictorTypes(Enum):
    STANDARD = 0


def create_predictor(t: PredictorTypes, params: PredictorParameters) -> LayoutAnalysisPredictor:
    if t == PredictorTypes.PIXEL_CLASSIFIER:
        from omr.layout.standard.predictor import StandardLayoutAnalysisPredictor
        return StandardLayoutAnalysisPredictor(params)

    raise Exception('Invalid type {}'.format(type))
