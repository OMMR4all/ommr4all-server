from abc import ABC, abstractmethod
from typing import NamedTuple, List, Optional, Generator
import numpy as np

from database import DatabasePage
from database.file_formats import PcGts
from database.file_formats.pcgts import Page
from database.file_formats.pcgts.page import Syllable, Annotations, SymbolType, GraphicalConnectionType
from database.file_formats.pcgts.page.annotations import Connection, SyllableConnector
from omr.dataset import RegionLineMaskData
from omr.steps.algorithm import AlgorithmPredictionResult, AlgorithmPredictor, PredictionCallback, \
    AlgorithmPredictionResultGenerator
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.predictor import SingleLinePredictionResult as TextSingleLinePredictionResult, Line


class SyllableMatchResult(NamedTuple):
    xpos: float
    syllable: Syllable

    def to_dict(self):
        return {
            'xPos': self.xpos,
            'syllable': self.syllable.to_json(),
        }


class MatchResult(NamedTuple):
    syllables: List[SyllableMatchResult]
    text_prediction: TextSingleLinePredictionResult
    text_line: Line
    music_line: Line

    def to_dict(self):
        return {
            'syllables': [s.to_dict() for s in self.syllables],
        }


class PageMatchResult(NamedTuple):
    match_results: List[MatchResult]
    pcgts: PcGts

    def pcgts(self):
        return self.pcgts

    def page(self) -> Page:
        return self.pcgts.page


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    annotations: List[Annotations]
    page_match_result: PageMatchResult

    def pcgts(self):
        return self.page_match_result.pcgts

    def page(self) -> Page:
        return self.pcgts().page

    def ds_page(self) -> DatabasePage:
        return self.page().location

    def to_dict(self):
        return {
            'annotations': [m.to_json() for m in self.annotations],
            'page': self.ds_page().page,
            'book': self.ds_page().book.book,
        }

    def store_to_page(self):
        page = self.page()
        page.annotations.connections.clear()
        raise NotImplementedError()


class SyllablesPredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        if not page.pcgts():
            return True
        return len(page.pcgts().page.annotations.connections) == 0

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        for pr in self._predict(pages, callback):
            yield PredictionResult(
                page_match_result=pr,
                annotations=[self._match_syllables_to_symbols(mr, pr.page()) for mr in pr.match_results]
            )

    @abstractmethod
    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[PageMatchResult, None, None]:
        pass

    def _match_syllables_to_symbols(self, mr: MatchResult, page: Page) -> Annotations:
        annotations = Annotations(page)

        max_d = np.mean([s2.xpos - s1.xpos for s1, s2 in zip(mr.syllables, mr.syllables[1:])])

        neumes = [s for s in mr.music_line.symbols if s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
        syllables_of_neumes = [[] for _ in neumes]

        def find_closest_neume(pos):
            closest = None
            d = 100000
            for s in neumes:
                dn = abs(pos - s.coord.x)
                if dn < d or not closest:
                    d = dn
                    closest = s

            return closest

        for s in mr.syllables:
            syllables_of_neumes[neumes.index(find_closest_neume(s.xpos))].append(s)

        for i, (n, sylls) in enumerate(zip(neumes, syllables_of_neumes)):
            if len(sylls) <= 1:
                continue

            sylls.sort(key=lambda s: s.xpos)

            left = sylls[0]
            right = sylls[-1]

            # check if we can move a syllable to the left or right
            left_neume = neumes[i - 1] if i > 0 and len(syllables_of_neumes[i - 1]) == 0 else None
            right_neume = neumes[i + 1] if i < len(neumes) - 1 and len(syllables_of_neumes[i + 1]) == 0 else None

            if left_neume and right_neume:
                d_left = abs(left_neume.coord.x - left.xpos)
                d_right = abs(right_neume.coord.x - right.xpos)
                if d_left > d_right:
                    left_neume = None
                else:
                    right_neume = None

            if left_neume is None and right_neume is None:
                continue
            elif left_neume is None:
                syllables_of_neumes[i + 1].append(right)
                sylls.remove(right)
            elif right_neume is None:
                syllables_of_neumes[i - 1].append(left)
                sylls.remove(left)

        annotations.get_or_create_connection(
            page.block_of_line(mr.music_line),
            page.block_of_line(mr.text_line),
        ).syllable_connections.extend(
            sum([[SyllableConnector(s.syllable, n) for s in sn] for n, sn in zip(neumes, syllables_of_neumes)], [])
        )
        return annotations
