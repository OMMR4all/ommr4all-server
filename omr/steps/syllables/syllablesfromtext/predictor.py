import os
from typing import Optional, List, Tuple, NamedTuple, Generator
import numpy as np

from database import DatabasePage, DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Syllable
from database.model import Model, MetaId
from omr.steps.step import Step, AlgorithmTypes
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictionResultGenerator, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from omr.steps.syllables.syllablesfromtext.meta import Meta
from omr.steps.syllables.predictor import PredictionResult, SyllablesPredictor, MatchResult, SyllableMatchResult, \
    PageMatchResult
from omr.steps.text.predictor import PredictionResult as TextPredictionResult
from omr.steps.text.predictor import SingleLinePredictionResult as TextSingleLinePredictionResult
#from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoderParams
import unidecode
from difflib import SequenceMatcher
from prettytable import PrettyTable


class SyllablesFromTextPredictor(SyllablesPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
        from ommr4all.settings import BASE_DIR
        model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/fraktur/text_calamari/', meta.type()))
        print(model.path)
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        #settings.params.ctcDecoder.params.type = CTCDecoderParams.CTC_DEFAULT
        self.ocr_predictor = meta.create_predictor(settings)

    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[PageMatchResult, None, None]:
        if callback:
            callback.progress_updated(0, len(pages), 0)

        for i, r in enumerate(self.ocr_predictor.predict(pages)):
            ocr_r: TextPredictionResult = r
            match_r = [self.match_text(text_line_r) for text_line_r in ocr_r.text_lines if len(text_line_r.line.operation.text_line.sentence.syllables) > 0]

            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))

            yield PageMatchResult(
                text_prediction_result=ocr_r,
                match_results=match_r,
                pcgts=ocr_r.pcgts,
            )

    def match_text(self, r: TextSingleLinePredictionResult, debug=False) -> MatchResult:
        # 1. Match prediction and gt strings
        # 2. Detect insertions/deletions/replacements/equal parts
        # 3. Assign predicted chars according to results of 2. to original syllables
        # 4. Compute the average x pos
        # 5. return

        pred = [(t, pos) for t, pos in r.text if t not in ' -']
        syls = r.line.operation.text_line.sentence.syllables
        assert(len(syls) > 0)

        # remove all "noisy" chars: ligatures/whitespace, ... for better match results
        def clean_text(t) -> str:
            # ß -> s only, not ss
            t = t.replace('ß', 's')
            t = unidecode.unidecode(t)
            return t.replace(' ', '').replace('-', '').lower()

        # Match the two sequences best possible
        gt = clean_text("".join([s.text for s in syls]))
        pred_txt = clean_text("".join([t for t, pos in pred]))
        sm = SequenceMatcher(a=pred_txt, b=gt, autojunk=False, isjunk=False)

        if debug:
            pt = PrettyTable(list(range(len(sm.get_opcodes()))))
            pt.add_row([gt[gt_start:gt_end] for _, _, _, gt_start, gt_end in sm.get_opcodes()])
            pt.add_row([pred_txt[pred_start:pred_end] for _, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes()])
            pt.add_row([opcode for opcode, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes()])
            print(pt)

        matches = []
        for opcode, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes():
            for i in range(gt_start, gt_end):
                if opcode == 'equal':
                    matches.append([pred[pred_start + i - gt_start]])
                elif opcode == 'insert':
                    matches.append([])
                elif opcode == 'delete':
                    # ignore (additional letter)
                    # maybe add to left or right
                    pass
                elif opcode == 'replace':
                    rel = (i - gt_start) / (gt_end - gt_start)
                    rel_ = (i + 1 - gt_start) / (gt_end - gt_start)
                    j = int(rel * (pred_end - pred_start) + pred_start)
                    j_ = int(rel_ * (pred_end - pred_start) + pred_start)
                    matches.append(list(pred[j:j_]))

        if debug:
            pt = PrettyTable(list(range(len(gt))))
            pt.add_row(list(gt))
            pt.add_row(matches)
            print(pt)

        pos = 0
        out_matches = []
        for syl in syls:
            m = sum(matches[pos:pos+len(syl.text)], [])
            if len(m) == 0:
                x = -1
            else:
                x = np.mean([p for _, p in m])
            out_matches.append({'s': syl, 'x': x})
            pos += len(syl.text)

        # interpolate syllables without any match
        ix = np.array([(i, match['x']) for i, match in enumerate(out_matches) if match['x'] >= 0])
        x_pos = np.interp(range(len(out_matches)), ix[:, 0], ix[:, 1])

        return MatchResult(
            syllables=[SyllableMatchResult(
                xpos=x,
                syllable=match['s'],
            ) for match, x in zip(out_matches, x_pos)],
            text_line=r.line.operation.text_line,
            music_line=r.line.operation.page.closest_music_line_to_text_line(r.line.operation.text_line),
        )


if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    from ommr4all.settings import BASE_DIR
    from database.file_formats.pcgts import PageScaleReference
    import random
    import matplotlib.pyplot as plt
    from shared.pcgtscanvas import PcGtsCanvas
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [
            # DatabaseBook('Graduel_Part_1'),
            # DatabaseBook('Graduel_Part_2'),
            # DatabaseBook('Graduel_Part_3'),
        ])
    book = DatabaseBook('Paper_New_York')
    meta = Step.meta(AlgorithmTypes.SYLLABLES_FROM_TEXT)
    model = meta.best_model_for_book(book)
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()[:1]))
    for i, p in enumerate(ps):
        pmr = p.page_match_result
        canvas = PcGtsCanvas(pmr.pcgts.page, PageScaleReference.NORMALIZED_X2)
        canvas.draw(pmr.text_prediction_result.text_lines[4], color=(25, 150, 25), background=True)
        # canvas.draw(pmr.match_results)
        # canvas.draw(p.annotations)
        canvas.show()
