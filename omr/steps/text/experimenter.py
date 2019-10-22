from typing import List

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts.page import Annotations
from omr.experimenter.experimenter import Experimenter
from .predictor import PredictionResult
from prettytable import PrettyTable
import numpy as np


class TextExperimenter(Experimenter):
    def print_results(self, results, log):
        tp, fp, fn, p, r, f1, acc = zip(*results)

        p, dp = np.mean(p), np.std(p)
        r, dr = np.mean(r), np.std(r)
        f1, df1 = np.mean(f1), np.std(f1)
        acc, dacc = np.mean(acc), np.std(acc)

        pt = PrettyTable(["P", "R", "F1", "Acc"])
        pt.add_row([p, r, f1, acc])
        pt.add_row([dp, dr, df1, dacc])
        log.info(pt)

    @classmethod
    def extract_gt_prediction(cls, full_predictions: List[PredictionResult]):
        def flatten(x):
            return sum(sum(x, []), [])

        pred = [[[(sc.syllable.id, sc.note.id) for sc in c.syllable_connections] for c in p.annotations.connections] for p in full_predictions]
        gt = [[[(sc.syllable.id, sc.note.id) for sc in c.syllable_connections] for c in p.page().annotations.connections] for p in full_predictions]
        return flatten(gt), flatten(pred)

    @classmethod
    def output_prediction_to_book(cls, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions: List[PredictionResult]):
        for pcgts, pred in zip(output_pcgts, predictions):
            pcgts.page.annotations = Annotations.from_json(pred.annotations.to_json(), pcgts.page)

    @classmethod
    def evaluate(cls, predictions, evaluation_params, log):
        gt, pred = predictions
        tp = len([x for x in gt if x in pred])
        fp = len([x for x in pred if x not in gt])
        fn = len([x for x in gt if x not in pred])

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        acc = tp / len(gt) if len(gt) == len(pred) else -1

        result = tp, fp, fn, p, r, f1, acc
        pt = PrettyTable(['tp', 'fp', 'fn', 'P', 'R', 'F1', 'acc'])
        pt.add_row(result)
        log.debug(pt)

        return result
