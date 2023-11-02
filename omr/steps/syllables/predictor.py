from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, List, Optional, Generator
import numpy as np
import scipy

from database import DatabasePage
from database.file_formats import PcGts
from database.file_formats.pcgts import Page, MusicSymbol
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
def _match_syllables_to_symbols_greedy(mr: MatchResult, page: Page, annotations: Annotations):
    if len(mr.syllables) == 0 or mr.music_line is None or len(mr.music_line.symbols) == 0:
        # no syllables present
        return
    neumes = [s for s in mr.music_line.symbols if
              s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
    if len(neumes) == 0:
        return
    syls = mr.syllables
    syls2 = mr.text_line.sentence.syllables

    def find_closest_neume(pos, search_list=None):
        closest = None
        d = 100000
        ind1 = None
        for ind, s in enumerate(search_list if search_list else neumes):
            dn = abs(pos - s.coord.x)
            if dn < d or not closest:
                d = dn
                closest = s
                ind1 = ind
        return closest, ind1, d

    def_dict = defaultdict(list)
    for i in syls:
        c, ind, dist = find_closest_neume(i.xpos, neumes)
        def_dict[str(ind)].append((i, ind, dist))
    move = True
    while move:
        move = False
        for i in def_dict.keys():
            if len(def_dict[i]) > 1:
                dist = 0
                sym = None
                sym_ind = None
                ind_of_ss = None

                def get_relevant_notes(h, neumes, def_dict):
                    @dataclass
                    class NeumeRel():
                        neume: MusicSymbol = None
                        ind: int = None

                    relevant = []
                    h = int(h)
                    if h - 1 >= 0:
                        if str(h - 1) not in def_dict:
                            relevant.append(NeumeRel(neume=neumes[h - 1], ind=h - 1))
                    if h + 1 < len(neumes):
                        if str(h + 1) not in def_dict:
                            relevant.append(NeumeRel(neume=neumes[h + 1], ind=h + 1))
                    relevant.append(NeumeRel(neume=neumes[h], ind=h))
                    return relevant
                    pass

                relevant_neumes = get_relevant_notes(i, neumes, def_dict)
                if len(relevant_neumes) > 1:
                    move = True
                    relevant_syls = sorted([(e[0], e[2], ind3) for ind3, e in enumerate(def_dict[i])],
                                           key=lambda x: x[1])
                    # relevant_syls = []
                    relevant_syls = relevant_syls[:len(relevant_neumes)]
                    weight_matrix = np.zeros(shape=(len(relevant_syls), len(relevant_neumes)))
                    for ind, syl in enumerate(relevant_syls):
                        for ind2, t_neum in enumerate(relevant_neumes):
                            weight_matrix[ind, ind2] = abs(t_neum.neume.coord.x - syl[0].xpos)
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(weight_matrix)
                    results = []

                    @dataclass
                    class Container():
                        syl: Syllable
                        note: MusicSymbol
                        neumes_ind: int
                        container_ind: int

                    for s in range(len(row_ind)):
                        syl = relevant_syls[s][0]
                        b_neume = relevant_neumes[col_ind[s]].neume
                        index = relevant_neumes[col_ind[s]].ind
                        syl_index = relevant_syls[s][2]

                        results.append(Container(syl=syl, note=b_neume, neumes_ind=index, container_ind=syl_index))

                    for res in sorted(results, key=lambda r: r.container_ind, reverse=True):
                        if res.neumes_ind == int(i):
                            continue
                        else:
                            def_dict[str(res.neumes_ind)].append((res.syl, res.neumes_ind, 0))
                            def_dict[i].pop(res.container_ind)
                        pass
                    break

                """
                for ind, t in enumerate(def_dict[i]):
                    if t[2] > dist:
                        dist = t[2]
                        sym = t[0]
                        sym_ind = t[1]
                        ind_of_ss = ind
                if sym_ind - 1 > 0 and str(sym_ind - 1) not in def_dict:
                    other = list(range(len(def_dict[i])))
                    other.pop(ind_of_ss)
                    left = True
                    for it in other:
                        if syls2.index(sym.syllable) <syls2.index(def_dict[i][it][0].syllable):
                            pass
                        else:
                            left = False
                    if left:
                        def_dict[i].pop(ind_of_ss)
                        def_dict[str(sym_ind - 1)].append((sym, sym_ind - 1, dist))
                        move = True
                        break
                if sym_ind + 1 < len(neumes) and str(sym_ind + 1) not in def_dict:
                    other = list(range(len(def_dict[i])))
                    other.pop(ind_of_ss)
                    right = True
                    for it in other:
                        if syls2.index(sym.syllable) > syls2.index(def_dict[i][it][0].syllable):
                            pass
                        else:
                            right = False
                    if right:
                        def_dict[i].pop(ind_of_ss)
                        def_dict[str(sym_ind + 1)].append((sym, sym_ind + 1, dist))
                        move = True
                        break
                    pass
                """
    annotations.get_or_create_connection(
        page.block_of_line(mr.music_line),
        page.block_of_line(mr.text_line),
    ).syllable_connections.extend(
        sum([[SyllableConnector(t[0].syllable, neumes[t[1]]) for t in def_dict[i]] for i in def_dict.keys()], [])
    )





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
        self.pcgts().page.annotations.connections.clear()
        self.pcgts().page.annotations = self.annotations
        self.pcgts().to_file(self.ds_page().file('pcgts').local_path())


class SyllablesPredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        if not page.pcgts():
            return True
        return len(page.pcgts().page.annotations.connections) == 0

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        for pr in self._predict(pages, callback):
            annotations = Annotations(pr.page())
            for mr in pr.match_results:
                # self._match_syllables_to_symbols_bipartite_matching(mr, pr.page(), annotations)
                #self._match_syllables_to_symbols(mr, pr.page(), annotations)
                _match_syllables_to_symbols_greedy(mr, pr.page(), annotations)

            yield PredictionResult(
                page_match_result=pr,
                annotations=annotations,
            )

    @abstractmethod
    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[
        PageMatchResult, None, None]:
        pass

    def _match_syllables_to_symbols_bipartite_matching(self, mr: MatchResult, page: Page, annotations: Annotations):
        if len(mr.syllables) == 0 or mr.music_line is None:
            # no syllables present
            return
        neumes = [s for s in mr.music_line.symbols if
                  s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
        syls = mr.syllables
        weight_matrix = np.zeros(shape=(len(syls), len(neumes)))
        for ind, i in enumerate(syls):
            for ind2, t in enumerate(neumes):
                weight_matrix[ind, ind2] = abs(t.coord.x - i.xpos)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(weight_matrix)
        # for i in range(len(row_ind)):
        #    print("row: ", row_ind[i], "  col: ", col_ind[i], "  value: ", weight_matrix[i, col_ind[i]])
        #    print("row: ", syls[i].syllable.text, "  col: ", col_ind[i], "  value: ", weight_matrix[i, col_ind[i]])
        # d = neumes[col_ind[i]]

        annotations.get_or_create_connection(
            page.block_of_line(mr.music_line),
            page.block_of_line(mr.text_line),
        ).syllable_connections.extend(
            sum([[SyllableConnector(syls[i].syllable, neumes[col_ind[i]])] for i in range(len(row_ind))], [])
        )


    def _match_syllables_to_symbols(self, mr: MatchResult, page: Page, annotations: Annotations):
        if len(mr.syllables) == 0 or mr.music_line is None:
            # no syllables present
            return

        max_d = np.mean([s2.xpos - s1.xpos for s1, s2 in zip(mr.syllables, mr.syllables[1:])])

        neumes = [s for s in mr.music_line.symbols if
                  s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
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

                sylls.sort(key=lambda s: mr.syllables.index(s))

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
