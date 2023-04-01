import os
from collections import namedtuple, defaultdict

from database import DatabaseBook, DatabasePage
from database.file_formats.pcgts import *
import logging
from typing import List, Optional, Tuple, NamedTuple

from database.file_formats.pcgts.page import Connection
from omr.steps.postprocessing.meta import Meta
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings, \
    AlgorithmPredictorParams, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator
import multiprocessing

logger = logging.getLogger(__name__)


class LineResult(NamedTuple):
    line: Line
    symbols: List[MusicSymbol]
    def to_dict(self):
        return {'symbols': [s.to_json() for s in self.symbols],
                'id': self.line.id}

class Result(NamedTuple):
    m_region: Block
    line_s: List[LineResult]


class PageResult(NamedTuple):
    p_result: List[Result]
    pcgts: PcGts


def _process_single(args: Tuple[DatabasePage, AlgorithmPredictorParams]):
    page, settings = args
    pcgts = page.pcgts()
    annotations = pcgts.page.annotations
    p_result = []
    m_regions = defaultdict(list)
    #for i in pcgts.page.all_text_lines():
        #print(i.text())
    for i in annotations.connections:
        m_regions[i.music_region.id].append(i)
    for im in m_regions.keys():
        ann_list = m_regions[im]
        #print(im)

        # i: Connection = i
        # i = sorted(i, key= lambda d: d)
        #text_region = i.text_region
        music_region = ann_list[0].music_region
        ids = [i.note.id for an in ann_list for i in an.syllable_connections]
        text = [i.syllable.text for an in ann_list for i in an.syllable_connections]
        #print(text)
        #print(ids)
        #syl_connections = sorted(i.syllable_connections, key=lambda d: d.note.coord.x)
        #connection = syl_connections[0] if len(syl_connections) > 0 else None

        ind_c = 0
        line_result = []
        prev_note = None

        for l in music_region.lines:
            line_symbols_deepcopy = Line.from_json(l.to_json()).symbols
            line_symbols_deepcopy.sort(key=lambda s: s.coord.x)
            for ind, s in enumerate(line_symbols_deepcopy):
                if s.symbol_type == s.symbol_type.NOTE:
                    #if connection and s.id == connection.note.id and not terminate:
                    if s.id in ids:
                        index = ids.index(s.id)
                        #print(index)
                        #print(text[index])
                        #print(s.graphical_connection)
                        pass
                        """
                        ind_c += 1
                        if ind_c < len(syl_connections):
                            prev_note = connection.note
                            connection = syl_connections[ind_c]
                            print(connection.syllable.text)
                            if connection and connection.syllable.text == "tis":
                                print(connection.syllable.text)
                                pass
                            while prev_note == connection.note:
                                ind_c += 1
                                if ind_c < len(syl_connections):
                                    prev_note = connection.note
                                    connection = syl_connections[ind_c]
                                else:
                                    terminate = True
                                    break
                        
                        else:
                            terminate = True
                        """

                    elif s.graphical_connection == s.graphical_connection.NEUME_START:
                        s.graphical_connection = s.graphical_connection.GAPED
            line_result.append(LineResult(l, line_symbols_deepcopy))
        p_result.append(Result(m_region=music_region, line_s=line_result))
    return PageResult(p_result, pcgts)


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PostProcessingResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    page: DatabasePage
    result: PageResult

    def to_dict(self):
        return {'musicLines': [i.to_dict() for l in self.result.p_result for i in l.line_s]}

    def store_to_page(self):

        for block in self.result.p_result:
            for line_s in block.line_s:
                line = line_s.line
                line.symbols = line_s.symbols

        self.result.pcgts.to_file(self.page.file('pcgts').local_path())
        pass


class PostprocessingPredictor(AlgorithmPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        if callback:
            callback.progress_updated(0, len(pages), 0)
        for i, page in enumerate(pages):
            input = (page, self.params)
            result = _process_single(input)
            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))
            yield PostProcessingResult(page=page, result=result)
        #with multiprocessing.Pool(processes=4) as pool:
        #    inputs = [(p, self.params) for p in pages]
        #    print(len(inputs))
        #    for i, result in enumerate(pool.imap(_process_single, inputs)):
        #        page: DatabasePage = inputs[i][0]
        #        percentage = (i + 1) / len(pages)
        #        if callback:
        #            callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))
        #        yield PostProcessingResult(page=page, result=result)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True


if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    from database import DatabaseBook

    b = DatabaseBook('mulhouse_mass_transcription_copy')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()][9]
    pred = PostprocessingPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in [val_pcgts]]))
    #for i in ps:
    #    i.store_to_page()
