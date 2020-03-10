import os
from typing import Optional, List, Tuple, NamedTuple, Generator
import numpy as np

from database import DatabasePage, DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Syllable, SymbolType, GraphicalConnectionType
from database.model import Model, MetaId
from omr.steps.step import Step, AlgorithmTypes
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictionResultGenerator, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from omr.steps.syllables.syllablesfromtext.meta import Meta
from omr.steps.syllables.predictor import PredictionResult, SyllablesPredictor, MatchResult, SyllableMatchResult, \
    PageMatchResult
from omr.steps.text.predictor import PredictionResult as TextPredictionResult, BlockType
from omr.steps.text.predictor import SingleLinePredictionResult as TextSingleLinePredictionResult
import unidecode
from difflib import SequenceMatcher
from prettytable import PrettyTable


class SyllablesInOrderPredictor(SyllablesPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[PageMatchResult, None, None]:
        if callback:
            callback.progress_updated(0, len(pages), 0)

        for i, page in enumerate(pages):
            pcgts = page.pcgts()

            mrs = []
            for tl in pcgts.page.lines_of_type(BlockType.LYRICS):
                ml = pcgts.page.closest_music_line_to_text_line(tl)
                neumes = [s for s in ml.symbols if s.symbol_type == SymbolType.NOTE and s.graphical_connection == GraphicalConnectionType.NEUME_START]
                neumes.sort(key=lambda n: n.coord.x)
                mrs.append(MatchResult(
                    syllables=[SyllableMatchResult(
                        xpos=neume.coord.x,
                        syllable=syl,
                    ) for syl, neume in zip(tl.sentence.syllables, neumes)],
                    text_line=tl,
                    music_line=ml,
                ))

            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))

            yield PageMatchResult(
                match_results=mrs,
                text_prediction_result=None,
                pcgts=pcgts,
            )


if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    from ommr4all.settings import BASE_DIR
    import random
    import matplotlib.pyplot as plt
    from shared.pcgtscanvas import PcGtsCanvas
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    from database.file_formats.pcgts import PageScaleReference
    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [
            # DatabaseBook('Graduel_Part_1'),
            # DatabaseBook('Graduel_Part_2'),
            # DatabaseBook('Graduel_Part_3'),
        ])
    book = DatabaseBook('Paper_New_York')
    meta = Step.meta(AlgorithmTypes.SYLLABLES_IN_ORDER)
    model = meta.best_model_for_book(book)
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()[:1]))
    for i, p in enumerate(ps):
        pmr = p.page_match_result
        canvas = PcGtsCanvas(pmr.pcgts.page, PageScaleReference.NORMALIZED_X2)
        canvas.draw(pmr.match_results)
        canvas.draw(p.annotations)
        canvas.show()
