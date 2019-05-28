from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple, Tuple
from database.file_formats.pcgts import *
from omr.text.dataset import TextDatasetParams, RegionLineMaskData, TextDataset
from enum import Enum


class TextPredictorParameters(NamedTuple):
    checkpoints: List[str]
    text_predictor_params: TextDatasetParams = TextDatasetParams()


class PredictionResult(NamedTuple):
    text: List[Tuple[str, Point]]
    line: RegionLineMaskData


PredictionType = Generator[PredictionResult, None, None]


class TextPredictor(ABC):
    def __init__(self, params: TextPredictorParameters):
        self.params = params
        self.dataset: TextDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        self.dataset = TextDataset(pcgts_files, self.params.text_predictor_params)
        return self._predict(self.dataset)

    @abstractmethod
    def _predict(self, dataset: TextDataset) -> PredictionType:
        pass


class PredictorTypes(Enum):
    CALAMARI = 0

    def __str__(self):
        return self.name


def create_predictor(t: PredictorTypes, params: TextPredictorParameters) -> TextPredictor:
    if t == PredictorTypes.CALAMARI:
        from omr.text.calamari.predictor import CalamariPredictor
        return CalamariPredictor(params)

    raise Exception('Invalid type {}'.format(type))
