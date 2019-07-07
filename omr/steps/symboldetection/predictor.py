from typing import List, Generator, NamedTuple
from database.file_formats.pcgts import *
from omr.dataset import RegionLineMaskData


class PredictionResult(NamedTuple):
    symbols: List[MusicSymbol]
    line: RegionLineMaskData


PredictionType = Generator[PredictionResult, None, None]
