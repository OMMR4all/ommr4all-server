from typing import Optional, List, Tuple, NamedTuple
import numpy as np

from database import DatabasePage
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Syllable
from database.model import Model, MetaId
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictionResultGenerator, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from omr.steps.syllables.syllablesfromtext.meta import Meta
from omr.steps.syllables.predictor import PredictionResult
from omr.steps.text.predictor import PredictionResult as TextPredictionResult
from omr.steps.text.predictor import SingleLinePredictionResult as TextSingleLinePredictionResult


class SyllableMatchResult(NamedTuple):
    xpos: float
    syllable: Syllable
    text_line: Line

class MatchResult(NamedTuple):
    syllables: List[SyllableMatchResult]


class PreprocessingPredictor(AlgorithmPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
        from ommr4all.settings import BASE_DIR
        model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical', meta.type()))
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        self.ocr_predictor = meta.create_predictor(settings)

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        if callback:
            callback.progress_updated(0, len(pages), 0)

        for i, r in enumerate(self.ocr_predictor.predict(pages)):
            ocr_r: TextPredictionResult = r
            match_r = [self.match_text(text_line_r) for text_line_r in ocr_r.text_lines]

            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))

            yield r


    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        if not page.pcgts():
            return True
        return len(page.pcgts().page.annotations.connections) == 0

    def match_text(self, r: TextSingleLinePredictionResult) -> MatchResult:
        pred = r.text
        syls = r.line.operation.text_line.sentence.syllables
        matches = [(s, pred[(i * len(pred)) // len(syls) : ((i+1) * len(pred)) // len(syls)]) for i, s in enumerate(syls)]

        return MatchResult(
            syllables=[SyllableMatchResult(
                xpos=float(np.mean([c.x for t, c in pred])),
                syllable=syl,
                text_line=r.line.operation.text_line,
            ) for syl, pred in matches]
        )



if __name__ == '__main__':
    from omr.steps.step import Step, AlgorithmTypes
    from ommr4all.settings import BASE_DIR
    import random
    import matplotlib.pyplot as plt
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [
            # DatabaseBook('Graduel_Part_1'),
            # DatabaseBook('Graduel_Part_2'),
            # DatabaseBook('Graduel_Part_3'),
        ])
    book = DatabaseBook('New_York')
    meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
    # model = meta.newest_model_for_book(book)
    model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical', meta.type()))
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()))
    orig = np.array(ps[0].text_lines[0].line.line_image)
    f, ax = plt.subplots(len(ps), 1)
    for i, p in enumerate(ps):
        ax[i].imshow(p.text_lines[0].line.line_image)
        print("".join([t[0] for t in p.text_lines[0].text]))

    plt.show()
