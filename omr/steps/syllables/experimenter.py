from typing import List

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts.page import Annotations, GraphicalConnectionType
from omr.experimenter.experimenter import Experimenter
from .predictor import PredictionResult
from prettytable import PrettyTable
import numpy as np


class SyllablesExperimenter(Experimenter):
    @classmethod
    def print_results(cls, args, results, log):
        tp, fp, fn, p, r, f1, acc = zip(*results)

        p, dp = np.mean(p), np.std(p)
        r, dr = np.mean(r), np.std(r)
        f1, df1 = np.mean(f1), np.std(f1)
        acc, dacc = np.mean(acc), np.std(acc)

        pt = PrettyTable(["P", "R", "F1", "Acc"])
        pt.add_row([p, r, f1, acc])
        pt.add_row([dp, dr, df1, dacc])
        log.info(pt)

        if args.magic_prefix:
            print("{}{}".format(args.magic_prefix, ','.join(map(str, [p, dp, r, dr, f1, df1, acc, dacc]))))

    def extract_gt_prediction(self, full_predictions: List[PredictionResult]):
        def flatten(x):
            return sum(sum(x, []), [])

        pred, gt = [], []
        for p in full_predictions:
            notes = {
                c.music_region.id: sum([[s for s in line.symbols if s.graphical_connection == GraphicalConnectionType.NEUME_START] for line in c.music_region.lines], []) for c in p.annotations.connections
            }
            pred.append([[(p.page().location.page, c.music_region.id, c.text_region.id, sc.syllable.id, notes[c.music_region.id].index(sc.note)) for sc in c.syllable_connections] for c in p.annotations.connections])
            gt.append([[(p.page().location.page, c.music_region.id, c.text_region.id, sc.syllable.id, notes[c.music_region.id].index(sc.note)) for sc in c.syllable_connections] for c in p.page().annotations.connections])

        return flatten(gt), flatten(pred)

    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions: List[PredictionResult]):
        for pcgts, pred in zip(output_pcgts, predictions):
            pcgts.page.annotations = Annotations.from_json(pred.annotations.to_json(), pcgts.page)

    def evaluate(self, predictions, evaluation_params):
        render = False

        gt, pred = predictions

        # compute distances
        tps = [x for x in gt if x in pred]
        fps = [x for x in pred if x not in gt]
        fns = [x for x in gt if x not in pred]

        distances = []
        for e in pred:
            matching_s = next((g for g in gt if g[:4] == e[:4]), None)
            if not matching_s:
                continue
            d = e[-1] - matching_s[-1]
            distances.append(d)

        distances = np.array(distances)
        if render:
            import matplotlib.pyplot as plt
            labels, counts = np.unique(distances, return_counts=True)
            r = np.arange(distances.min(), distances.max() + 1)
            counts = [(counts[np.where(labels == i)] if i in labels else 0) for i in r]
            plt.bar(r, counts, align='center')
            # plt.gca().set_yscale('log')
            plt.gca().set_xticks(r)
            plt.grid(axis='y')
            plt.gca().set_xlabel('Distance to correct neume')
            plt.gca().set_ylabel('Count')
            plt.show()

        # compute f1 scores
        tp = len(tps)
        fp = len(fps)
        fn = len(fns)

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        acc = tp / len(gt) if len(gt) == len(pred) else -1

        result = tp, fp, fn, p, r, f1, acc
        pt = PrettyTable(['tp', 'fp', 'fn', 'P', 'R', 'F1', 'acc'])
        pt.add_row(result)
        self.fold_log.debug(pt)

        return result
