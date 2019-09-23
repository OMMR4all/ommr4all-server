from omr.experimenter.experimenter import Experimenter
from database import DatabaseBook
from omr.steps.symboldetection.dataset import PcGts
from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, Counts, precision_recall_f1, AccCounts
from typing import List
from prettytable import PrettyTable
import numpy as np


class SymbolsExperimenter(Experimenter):
    @classmethod
    def extract_gt_prediction(cls, full_predictions):
        return zip(*[(p.line.operation.music_line.symbols, p.symbols) for p in sum([p.music_lines for p in full_predictions], [])])

    @classmethod
    def output_prediction_to_book(cls, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
        output_pcgts_by_page_name = {}
        for o_pcgts in output_pcgts:
            output_pcgts_by_page_name[o_pcgts.page.location.page] = o_pcgts
            for mr in o_pcgts.page.music_blocks():
                for ml in mr.lines:
                    ml.symbols = []  # clear all symbols

            # clear all annotations
            o_pcgts.page.annotations.connections = []
            o_pcgts.page.comments.comments = []

        for p in predictions:
            o_pcgts = output_pcgts_by_page_name[p.line.operation.page.location.page]
            o_pcgts.page.music_line_by_id(p.line.operation.music_line.id).symbols = p.symbols

    @classmethod
    def evaluate(cls, predictions, evaluation_params, log):
        gt_symbols, pred_symbols = predictions
        evaluator = SymbolDetectionEvaluator(evaluation_params)
        metrics, counts, acc_counts, acc_acc, total_diffs = evaluator.evaluate(gt_symbols, pred_symbols)

        at = PrettyTable()

        at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])
        at.add_column("TP", counts[:, Counts.TP])
        at.add_column("FP", counts[:, Counts.FP])
        at.add_column("FN", counts[:, Counts.FN])

        prec_rec_f1 = np.array([precision_recall_f1(c[Counts.TP], c[Counts.FP], c[Counts.FN]) for c in counts])
        at.add_column("Precision", prec_rec_f1[:, 0])
        at.add_column("Recall", prec_rec_f1[:, 1])
        at.add_column("F1", prec_rec_f1[:, 2])
        log.debug(at.get_string())

        at = PrettyTable()
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("True", acc_counts[:, AccCounts.TRUE])
        at.add_column("False", acc_counts[:, AccCounts.FALSE])
        at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
        at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

        log.debug(at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
        at.add_row(total_diffs)
        log.debug(at)

        return prec_rec_f1, acc_acc, total_diffs

    def print_results(self, results, log):
        log.info("Total Result:")

        prec_rec_f1_list = [r[0] for r in results if r is not None]
        acc_counts_list = [r[1] for r in results if r is not None]
        total_diffs = [r[2] for r in results if r is not None]
        if len(prec_rec_f1_list) == 0:
            return

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        acc_mean = np.mean(acc_counts_list, axis=0)
        acc_std = np.std(acc_counts_list, axis=0)
        diffs_mean = np.mean(total_diffs, axis=0)
        diffs_std = np.std(total_diffs, axis=0)

        at = PrettyTable()

        at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])

        at.add_column("Precision", prf1_mean[:, 0])
        at.add_column("+-", prf1_std[:, 0])
        at.add_column("Recall", prf1_mean[:, 1])
        at.add_column("+-", prf1_std[:, 1])
        at.add_column("F1", prf1_mean[:, 2])
        at.add_column("+-", prf1_std[:, 2])
        log.info("\n" + at.get_string())

        at = PrettyTable()
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("Accuracy [%]", acc_mean[:, 0] * 100)
        at.add_column("+- [%]", acc_std[:, 0] * 100)

        log.info("\n" + at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
        at.add_row(diffs_mean)
        at.add_row(diffs_std)
        log.info("\n" + at.get_string())

        if self.global_args.magic_prefix:
            # skip first all output
            all_symbol_detection = np.array(sum([[prf1_mean[1:, i], prf1_std[1:, i]] for i in range(3)], [])).transpose().reshape(-1)
            all_acc = np.array(np.transpose([acc_mean[:, 0], acc_std[:, 0]]).reshape([-1]))
            all_diffs = np.array(np.transpose([diffs_mean, diffs_std])).reshape([-1])
            print("{}{}".format(self.global_args.magic_prefix, ','.join(map(str, list(all_symbol_detection) + list(all_acc) + list(all_diffs)))))
