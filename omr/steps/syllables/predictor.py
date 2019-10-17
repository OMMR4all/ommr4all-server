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
from omr.steps.text.predictor import PredictionResult as TextPredictionResult


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
    text_line: Line
    music_line: Line

    def to_dict(self):
        return {
            'syllables': [s.to_dict() for s in self.syllables],
        }


class PageMatchResult(NamedTuple):
    match_results: List[MatchResult]
    text_prediction_result: Optional[TextPredictionResult]
    pcgts: PcGts

    def pcgts(self):
        return self.pcgts

    def page(self) -> Page:
        return self.pcgts.page


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    annotations: Annotations
    page_match_result: PageMatchResult

    def pcgts(self):
        return self.page_match_result.pcgts

    def page(self) -> Page:
        return self.pcgts().page

    def ds_page(self) -> DatabasePage:
        return self.page().location

    def to_dict(self):
        return {
            'annotations': self.annotations.to_json(),
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
            annotations = Annotations(pr.page())
            for mr in pr.match_results:
                self._match_syllables_to_symbols(mr, pr.page(), annotations)
            yield PredictionResult(
                page_match_result=pr,
                annotations=annotations,
            )

    @abstractmethod
    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[PageMatchResult, None, None]:
        pass

    def _match_syllables_to_symbols(self, mr: MatchResult, page: Page, annotations: Annotations):
        if len(mr.syllables) == 0:
            # no syllables present
            return

        max_d = np.mean([s2.xpos - s1.xpos for s1, s2 in zip(mr.syllables, mr.syllables[1:])])

        neumes = [s for s in mr.music_line.symbols if s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
        avg_neume_distance = mr.music_line.avg_neume_distance(None)
        if avg_neume_distance is None:
            return

        syllables_of_neumes = [[] for _ in neumes]

        def find_closest_neume(pos, search_list=None):
            closest = None
            d = 100000
            for s in (search_list if search_list else neumes):
                dn = abs(pos - s.coord.x)
                if dn < d or not closest:
                    d = dn
                    closest = s

            return closest

        def find_closest_syllable(pos):
            closest = None
            d = 100000
            for s in mr.syllables:
                dn = abs(pos - s.xpos)
                if dn < d or not closest:
                    d = dn
                    closest = s

            return closest

        neumes_of_syllables = [[] for _ in mr.syllables]
        for n in neumes:
            neumes_of_syllables[mr.syllables.index(find_closest_syllable(n.coord.x))].append(n)

        for nos, syl in zip(neumes_of_syllables, mr.syllables):
            if len(nos) == 0:
                continue

            nos.sort(key=lambda n: n.coord.x)

            neumes_to_keep = [find_closest_neume(syl.xpos, nos)]

            # check how many prev neumes to keep (notes after the following are irrelevant)
            idx = nos.index(neumes_to_keep[0])
            for i in reversed(range(0, idx)):
                n = nos[i]
                next = nos[i + 1]
                last_n = mr.music_line.last_note_of_neume(n)
                if abs(last_n.coord.x - next.coord.x) > avg_neume_distance:
                    break

                neumes_to_keep.insert(0, n)

            best_n = neumes_to_keep[0]
            syllables_of_neumes[neumes.index(best_n)].append(syl)

        for unassigned in [s for s in mr.syllables if not any([s in sylls for sylls in syllables_of_neumes])]:
            syllables_of_neumes[neumes.index(find_closest_neume(unassigned.xpos))].append(unassigned)

        moved = True
        while moved:
            moved = False
            for i, (n, sylls) in enumerate(zip(neumes, syllables_of_neumes)):
                if len(sylls) <= 1:
                    continue

                sylls.sort(key=lambda s: s.xpos)

                left = sylls[0]
                right = sylls[-1]

                def try_move(n_i, direction):
                    if len(syllables_of_neumes[n_i]) == 0:
                        return n_i

                    try:
                        next_syllables = syllables_of_neumes[n_i + direction]
                    except IndexError:
                        return -1
                    if len(next_syllables) == 0:
                        return n_i

                    return try_move(n_i + direction, direction)

                def move(syll, start, until, direction):
                    if start < until:
                        cur = syllables_of_neumes[start + 1][0]
                        move(cur, start + 1, until, direction)

                    next = syllables_of_neumes[start + direction]
                    cur = syllables_of_neumes[start]
                    cur.remove(syll)
                    next.append(syll)

                r = try_move(i, +1)
                if r >= 0:
                    move(right, i, r, +1)
                    moved = True
                else:
                    l = try_move(i, -1)
                    if l >= 0:
                        move(left, i, r, -1)
                        moved = True

        annotations.get_or_create_connection(
            page.block_of_line(mr.music_line),
            page.block_of_line(mr.text_line),
        ).syllable_connections.extend(
            sum([[SyllableConnector(s.syllable, n) for s in sn] for n, sn in zip(neumes, syllables_of_neumes)], [])
        )
