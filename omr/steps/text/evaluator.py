from typing import NamedTuple, List, Tuple
import re

from database.file_formats.pcgts.page import Sentence


class EvaluatorParams(NamedTuple):
    re_ignore: str = r'[ \-]'


class Evaluator:
    def __init__(self, params: EvaluatorParams):
        self.params = params

    def evaluate(self, gt_sentences: List[Sentence], preds_text_pos: List[List[Tuple[str, int]]]):
        from calamari_ocr.ocr.evaluator import Evaluator

        gt = [s.text() for s in gt_sentences]
        pred = ["".join([s for s, _ in p]) for p in preds_text_pos]

        if self.params.re_ignore:
            re_ignore = re.compile(self.params.re_ignore)
            gt = [re_ignore.sub('', t) for t in gt]
            pred = [re_ignore.sub('', t) for t in pred]

        r = Evaluator.evaluate(gt_data=gt, pred_data=pred)
        return r
