from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple, Tuple, Optional

from database import DatabasePage
from database.file_formats.pcgts import *
from database.file_formats.pcgts.page import Sentence
from omr.dataset import RegionLineMaskData

from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictionResultGenerator, \
    AlgorithmPredictionResult
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.dataset import TextDataset


class SingleLinePredictionResult(NamedTuple):
    text: List[Tuple[str, Point]]
    line: RegionLineMaskData
    hyphenated: str
    chars: List[Tuple[str, List[Point]]] = None

    def to_dict(self):
        return {'sentence': self.hyphenated,
                'id': self.line.operation.text_line.id,
                }


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    pcgts: PcGts
    dataset_page: DatabasePage
    text_lines: List[SingleLinePredictionResult]

    def to_dict(self):
        return {'textLines': [l.to_dict() for l in self.text_lines]}

    def store_to_page(self):
        #print(self.dataset_page.file('pcgts').local_path())
        for line in self.text_lines:
            line.line.operation.text_line.sentence = Sentence.from_string(line.hyphenated)
        self.pcgts.page.annotations.connections.clear()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class TextPredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return all([len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines()])

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        dataset = TextDataset(pcgts_files, self.dataset_params)
        page_results = [PredictionResult(pcgts, pcgts.page.location, []) for pcgts in pcgts_files]
        for line_results in self._predict(dataset, callback):
            page_result = [p for p in page_results if p.pcgts == line_results.line.operation.pcgts][0]
            page_result.text_lines.append(line_results)

        for page_result in page_results:
            yield page_result

    @abstractmethod
    def _predict(self, dataset: TextDataset, callback: Optional[PredictionCallback]) -> Generator[SingleLinePredictionResult, None, None]:
        pass
