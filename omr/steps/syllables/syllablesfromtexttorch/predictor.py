import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, NamedTuple, Generator

import edlib
import numpy as np

from database import DatabasePage, DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import Line, Point
from database.file_formats.pcgts.page import Syllable
from database.model import Model, MetaId
from omr.steps.step import Step, AlgorithmTypes
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictionResultGenerator, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings, AlgorithmPredictorParams
from omr.steps.syllables.syllablesfromtexttorch.meta import Meta
from omr.steps.syllables.predictor import PredictionResult, SyllablesPredictor, MatchResult, SyllableMatchResult, \
    PageMatchResult
from omr.steps.text.predictor import PredictionResult as TextPredictionResult
from omr.steps.text.predictor import SingleLinePredictionResult as TextSingleLinePredictionResult
# from calamari_ocr.ocr.backends.ctc_decoder.ctc_decoder import CTCDecoderParams
import unidecode
from difflib import SequenceMatcher
from prettytable import PrettyTable


class SyllablesFromTextPredictor(SyllablesPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        meta = Step.meta(AlgorithmTypes.OCR_GUPPY)
        from ommr4all.settings import BASE_DIR
        model = Model(
            MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/french14/text_guppy/', meta.type()))
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        # settings.params.ctcDecoder.params.type = CTCDecoderParams.CTC_DEFAULT
        self.ocr_predictor = meta.create_predictor(settings)

    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> Generator[
        PageMatchResult, None, None]:
        if callback:
            callback.progress_updated(0, len(pages), 0)

        for i, r in enumerate(self.ocr_predictor.predict(pages, callback=callback)):
            ocr_r: TextPredictionResult = r
            # try:
            match_r = [self.match_text2(text_line_r) for text_line_r in ocr_r.text_lines if
                       len(text_line_r.line.operation.text_line.sentence.syllables) > 0]

            # match_r = [self.match_text(text_line_r) for text_line_r in ocr_r.text_lines if len(text_line_r.line.operation.text_line.sentence.syllables) > 0]
            # except Exception as e:
            # print(e)
            # match_r = []
            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))

            yield PageMatchResult(
                text_prediction_result=ocr_r,
                match_results=match_r,
                pcgts=ocr_r.pcgts,
            )

    def match_text2(self, r: TextSingleLinePredictionResult, debug=False) -> MatchResult:
        # 1. Match prediction and gt strings
        # 2. Detect insertions/deletions/replacements/equal parts
        # 3. Assign predicted chars according to results of 2. to original syllables
        # 4. Compute the average x pos
        # 5. return

        pred = [(t, pos) for t, pos in r.text if t not in ' -']
        syls = r.line.operation.text_line.sentence.syllables
        # print(pred)
        # print([i.text for i in syls])
        assert (len(syls) > 0)

        # remove all "noisy" chars: ligatures/whitespace, ... for better match results
        def clean_text(t) -> str:
            # ß -> s only, not ss
            t = t.replace('ß', 's')
            t = t.replace('v', 'u')
            t = t.replace('ſ', 's')
            t = unidecode.unidecode(t)
            return t.replace(' ', '').replace('-', '').lower()

        # Match the two sequences best possible
        gt = clean_text("".join([s.text for s in syls]))
        pred_txt = clean_text("".join([t for t, pos in pred]))

        @dataclass
        class CharPosContainer():
            char: str
            xpos: Point = None

            def __hash__(self):
                return hash(self.char)

        gt = [CharPosContainer(i) for i in gt]
        pred_txt = [CharPosContainer(clean_text(t), xpos=pos) for t, pos in pred]
        gap_symbol = "-"
        #ed = edlib.align(pred_txt, gt, mode="NW", task="path")
        inp = "".join([i.char for i in pred_txt])
        gtp = "".join([i.char for i in gt])
        if len(inp) == 0:
            inp += " "
        if len(gtp) == 0:
            gtp += " "
        ed = edlib.align(inp, gtp, mode="NW", task="path")


        #print(f"input: {inp}")
        #print(f"inpus: {gtp}")

        nice = edlib.getNiceAlignment(ed, inp, gtp,
                                      gapSymbol=gap_symbol)
        #print(nice["matched_aligned"])
        for ind, i in enumerate(nice["query_aligned"]):
            if i == gap_symbol:
                pred_txt.insert(ind, CharPosContainer(gap_symbol))

        for ind, i in enumerate(nice["target_aligned"]):
            if i == gap_symbol:
                gt.insert(ind, CharPosContainer(gap_symbol))
        for gt_char, op, p in zip(gt, nice["matched_aligned"], pred_txt):
            if op == gap_symbol:
                #print(gt_char)
                #print(p.xpos)
                #print(gt_char)

                continue
            else:
                #print(gt_char)
                #print(p.xpos)
                gt_char.xpos = p.xpos
                #print(gt_char)
        #print(nice["matched_aligned"])
        gt = [i for i in gt if i.char != gap_symbol]

        out_matches = []
        pos = 0

        for syl in syls:
            #print( len(syl.text))
            #m = sum([i.xpos for i in gt[pos:pos + len(syl.text)] if i.xpos is not None], [])
            m = [i.xpos for i in gt[pos:pos + len(syl.text)] if i.xpos is not None]
            #print(gt[pos:pos + len(syl.text)])
            #print(m)
            if len(m) == 0:
                x = -1
            else:
                x = np.mean(m)
                #x = m[0]
            out_matches.append({'s': syl, 'x': x})
            pos += len(syl.text)
        if len(pred) > 0:
            # interpolate syllables without any match
            ix = np.array([(i, match['x']) for i, match in enumerate(out_matches) if match['x'] >= 0])
            #print(ix)
            x_pos = np.interp(range(len(out_matches)), ix[:, 0], ix[:, 1])
        else:
            x_pos = np.linspace(0, 1, len(out_matches), endpoint=False)

        # for m, x in zip(out_matches, x_pos):
        #    print(f'Text: {m["s"].text}, Pos: {str(x)}, Pos2: {str(m["x"])}' )
        return MatchResult(
            syllables=[SyllableMatchResult(
                xpos=x,
                syllable=match['s'],
            ) for match, x in zip(out_matches, x_pos)],
            text_line=r.line.operation.text_line,
            music_line=r.line.operation.page.closest_music_line_to_text_line(r.line.operation.text_line),
        )

    def match_text(self, r: TextSingleLinePredictionResult, debug=False) -> MatchResult:
        # 1. Match prediction and gt strings
        # 2. Detect insertions/deletions/replacements/equal parts
        # 3. Assign predicted chars according to results of 2. to original syllables
        # 4. Compute the average x pos
        # 5. return

        pred = [(t, pos) for t, pos in r.text if t not in ' -']
        syls = r.line.operation.text_line.sentence.syllables
        # print(pred)
        # print([i.text for i in syls])
        assert (len(syls) > 0)

        # remove all "noisy" chars: ligatures/whitespace, ... for better match results
        def clean_text(t) -> str:
            # ß -> s only, not ss
            t = t.replace('ß', 's')
            t = t.replace('v', 'u')
            t = t.replace('ſ', 's')
            t = unidecode.unidecode(t)
            return t.replace(' ', '').replace('-', '').lower()

        # Match the two sequences best possible
        gt = clean_text("".join([s.text for s in syls]))
        pred_txt = clean_text("".join([t for t, pos in pred]))
        sm = SequenceMatcher(a=pred_txt, b=gt, autojunk=False, isjunk=False)
        # print(f'GT1: {r.line.operation.text_line.sentence.text()}')
        # print(f'PR1: {r.hyphenated}')

        # print(f'GT: {gt}')
        # print(f'PR: {pred_txt}')
        if debug:
            pt = PrettyTable(list(range(len(sm.get_opcodes()))))
            pt.add_row([gt[gt_start:gt_end] for _, _, _, gt_start, gt_end in sm.get_opcodes()])
            pt.add_row(
                [pred_txt[pred_start:pred_end] for _, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes()])
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
            m = sum(matches[pos:pos + len(syl.text)], [])
            if len(m) == 0:
                x = -1
            else:
                x = np.mean([p for _, p in m])
            out_matches.append({'s': syl, 'x': x})
            pos += len(syl.text)

        if len(pred) > 0:
            # interpolate syllables without any match
            ix = np.array([(i, match['x']) for i, match in enumerate(out_matches) if match['x'] >= 0])
            x_pos = np.interp(range(len(out_matches)), ix[:, 0], ix[:, 1])
        else:
            x_pos = np.linspace(0, 1, len(out_matches), endpoint=False)

        # for m, x in zip(out_matches, x_pos):
        #    print(f'Text: {m["s"].text}, Pos: {str(x)}, Pos2: {str(m["x"])}' )
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
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True),
                                                               LockState(Locks.LAYOUT, True)], True, [
                                                             # DatabaseBook('Graduel_Part_1'),
                                                             # DatabaseBook('Graduel_Part_2'),
                                                             # DatabaseBook('Graduel_Part_3'),
                                                         ])
    book = DatabaseBook('mul_2_rsync_base_symbol_finetune_w_pp')
    meta = Step.meta(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH)
    model = meta.best_model_for_book(book)
    # model = Model.from_id_str()
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()[92:93]))
    for i, p in enumerate(ps):
        pmr = p.page_match_result
        canvas = PcGtsCanvas(pmr.pcgts.page, PageScaleReference.NORMALIZED_X2)
        canvas.draw(pmr.match_results, color=(25, 150, 25), background=True)
        # canvas.draw(pmr.match_results)
        # canvas.draw(p.annotations)
        canvas.show()
