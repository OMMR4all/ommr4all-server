import logging
from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import Block, BlockType
from typing import List
from prettytable import PrettyTable
import numpy as np

from omr.experimenter.experimenter import Experimenter
from omr.steps.stafflines.detection.evaluator import EvaluationData, StaffLineDetectionEvaluator

logger = logging.getLogger(__name__)


class StaffLinesExperimenter(Experimenter):
    def extract_gt_prediction(self, full_predictions):
        return [
            EvaluationData(
                p.line.operation.page.location.local_path(),
                p.line.operation.music_lines,
                p.music_lines,
                p.line.operation.page_image.shape,
            )
            for p in full_predictions
        ]

    def evaluate(self, predictions, evaluation_params):
        evaluator = StaffLineDetectionEvaluator(evaluation_params)
        res = evaluator.evaluate(predictions)
        counts, prf1, (all_tp_staves, all_fp_staves, all_fn_staves) = res
        if counts.shape[0] > 0:
            at = PrettyTable()

            at.add_column("Type", ["Staff lines found", "Staff lines hit", "Staves found", "Staff lines hit"])
            at.add_column("TP", counts[:, 0])
            at.add_column("FP", counts[:, 1])
            at.add_column("FN", counts[:, 2])
            at.add_column("Total", counts[:, 3])

            at.add_column("Precision", prf1[:, 0])
            at.add_column("Recall", prf1[:, 1])
            at.add_column("F1", prf1[:, 2])
            self.fold_log.debug(at.get_string())

            metrics = prf1
        else:
            self.fold_log.warning("Empty file without ground truth lines")
            metrics = None

        return res

    @classmethod
    def print_results(cls, args, results, log):
        counts, metrics, _ = zip(*results)

        logger.info("Total Result:")

        at = PrettyTable(["Fold", "TP", "FP", "FN", "Total", "Precision", "Recall", "F1"])
        for id, (c, m, _) in enumerate(results):
            at.add_row([id] + list(c[0]) + list(m[0]))

        logger.info("\n\nStaff lines detected:\n" + at.get_string())

        at = PrettyTable(["Fold", "TP", "FP", "FN", "Total", "Precision", "Recall", "F1"])
        for id, (c, m, _) in enumerate(results):
            at.add_row([id] + list(c[1]) + list(m[1]))

        logger.info("\n\nStaff lines hit:\n" + at.get_string())

        prec_rec_f1_list = metrics

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        at = PrettyTable()

        at.add_column("Type", ["Staff Lines Detected", "Staff lines hit", "Staves found", "Staff lines hit"])

        at.add_column("Precision", prf1_mean[:, 0])
        at.add_column("+-", prf1_std[:, 0])
        at.add_column("Recall", prf1_mean[:, 1])
        at.add_column("+-", prf1_std[:, 1])
        at.add_column("F1", prf1_mean[:, 2])
        at.add_column("+-", prf1_std[:, 2])
        logger.info("\n\n" + at.get_string())

        if args.magic_prefix:
            all_values = np.array(sum([[prf1_mean[:, i], prf1_std[:, i]] for i in range(3)], [])).transpose().reshape(-1)
            print("{}{}".format(args.magic_prefix, ','.join(map(str, all_values))))

    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
        output_pcgts_by_page_name = {}
        for o_pcgts in output_pcgts:
            output_pcgts_by_page_name[o_pcgts.page.location.page] = o_pcgts

            # clear page
            o_pcgts.page.music_regions = []
            o_pcgts.page.text_regions = []
            o_pcgts.page.annotations.connections = []
            o_pcgts.page.comments.comments = []

        for p in predictions:
            o_pcgts = output_pcgts_by_page_name[p.line.operation.page.location.page]
            for ml in p.music_lines:
                o_pcgts.page.music_regions.append(
                    Block(block_type=BlockType.MUSIC, lines=[ml])
                )

        for o_pcgts in output_pcgts:
            o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())

    def output_book_2(self, test_pcgts_files, predictions, all_tp_staves, all_fp_staves, all_fn_staves):
        global_args = self.args.global_args
        output_tp = True
        output_fp = True
        output_fn = True
        if global_args.output_only_tp:
            output_fp = False
            output_fn = False

        if global_args.output_only_fp:
            output_tp = False
            output_fn = False

        logger.info("Outputting music lines to {}".format(global_args.output_book))
        assert (predictions is not None)
        pred_book = DatabaseBook(global_args.output_book)
        output_pcgts_by_page_name = {}
        for pcgts in test_pcgts_files:
            o_pcgts = PcGts.from_file(pred_book.page(pcgts.page.location.page).file('pcgts'))
            output_pcgts_by_page_name[pcgts.page.location.page] = o_pcgts
            o_pcgts.page.music_regions.clear()

        for p, tp_staves, fp_staves, fn_staves in zip(predictions, all_tp_staves, all_fp_staves, all_fn_staves):
            o_pcgts = output_pcgts_by_page_name[p.line.operation.page.location.page]
            if output_tp:
                for ml, gt_ml in [(ml, gt_ml) for ml, gt_ml, _ in tp_staves if ml in p.music_lines]:
                    if global_args.output_symbols:
                        ml.symbols = gt_ml.symbols[:]
                    o_pcgts.page.music_regions.append(
                        Block(block_type=BlockType.MUSIC, lines=[ml])
                    )

            if output_fp:
                for ml in [ml for ml in fp_staves if ml in p.music_lines]:
                    ml.symbols.clear()
                    o_pcgts.page.music_regions.append(
                        Block(block_type=BlockType.MUSIC, lines=[ml])
                    )

            if output_fn:
                for gt_ml in fn_staves:
                    if not global_args.output_symbols:
                        gt_ml.symbols.clear()

                    o_pcgts.page.music_regions.append(
                        Block(block_type=BlockType.MUSIC, lines=[gt_ml])
                    )

        for _, o_pcgts in output_pcgts_by_page_name.items():
            o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())
