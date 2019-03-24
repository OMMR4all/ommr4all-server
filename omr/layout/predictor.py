from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple, Dict
from database.file_formats.pcgts import *
from omr.symboldetection.dataset import SymbolDetectionDataset
from enum import Enum


class LayoutPredictorParameters(NamedTuple):
    checkpoints: List[str]


class PredictionResult(NamedTuple):
    text_regions: Dict[TextRegionType, List[Coords]]
    music_regions: List[Coords]


PredictionType = Generator[PredictionResult, None, None]


class IdCoordsPair(NamedTuple):
    coords: Coords
    id: str = None

    def to_dict(self):
        return {
            'coords': self.coords.to_json(),
            'id': self.id,
        }


class FinalPredictionResult(NamedTuple):
    text_regions: Dict[TextRegionType, List[IdCoordsPair]]
    music_regions: List[IdCoordsPair]

    def to_dict(self):
        return {
            'textRegions': {
                key.value: [v.to_dict() for v in val] for key, val in self.text_regions.items()
            },
            'musicRegions': [v.to_dict() for v in self.music_regions]
        }


FinalPrediction = Generator[FinalPredictionResult, None, None]


class LayoutAnalysisPredictor(ABC):
    def __init__(self, params: LayoutPredictorParameters):
        self.params = params
        self.dataset: SymbolDetectionDataset = None

    def predict(self, pcgts_files: List[PcGts]) -> FinalPrediction:
        for r, pcgts in zip(self._predict(pcgts_files), pcgts_files):
            music_lines = []
            for mr in pcgts.page.music_regions:
                music_lines += mr.staffs

            # music lines must be sorted
            music_lines.sort(key=lambda ml: ml.center_y())

            for ml, coords in zip(music_lines, r.music_regions):
                ml.coords = coords

            yield FinalPredictionResult(
                {k: [IdCoordsPair(c) for c in coords] for k, coords in r.text_regions.items()},
                [IdCoordsPair(coords, str(ml.id)) for ml, coords in zip(music_lines, r.music_regions)]
            )

    @abstractmethod
    def _predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        pass


class PredictorTypes(Enum):
    STANDARD = 0


def create_predictor(t: PredictorTypes, params: LayoutPredictorParameters) -> LayoutAnalysisPredictor:
    if t == PredictorTypes.STANDARD:
        from omr.layout.standard.predictor import StandardLayoutAnalysisPredictor
        return StandardLayoutAnalysisPredictor(params)

    raise Exception('Invalid type {}'.format(type))
