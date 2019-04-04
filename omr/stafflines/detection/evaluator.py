from omr.stafflines.detection.predictor import StaffLinesPredictor, StaffLinePredictorParameters, StaffLineDetectionDatasetParams
from database.file_formats.pcgts import PcGts, StaffLines, StaffLine, MusicLines, MusicLine
from typing import List, NamedTuple, Tuple
import numpy as np
from omr.symboldetection.evaluator import precision_recall_f1
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


class EvaluationData(NamedTuple):
    label: str
    gt: MusicLines
    pred: MusicLines
    shape: Tuple[int, int]

    @staticmethod
    def from_json(d: dict):
        return EvaluationData(d['label'], MusicLines.from_json(d['gt']), MusicLines.from_json(d['pred']), tuple(d['shape']))

    def to_json(self):
        return {
            'label': self.label,
            'gt': self.gt.to_json(),
            'pred': self.pred.to_json(),
            'shape': list(self.shape)
        }


class EvaluationParams(NamedTuple):
    line_hit_overlap_threshold: float = 0.5
    staff_n_lines_threshold: int = 2
    staff_line_found_distance: int = 5
    debug: bool = False


class StaffLineDetectionEvaluator:
    def __init__(self, params=None):
        self.params = params if params else EvaluationParams()

    def evaluate(self, data: List[EvaluationData]):
        all_counts = np.zeros([0, 4, 4])
        all_prf1 = np.zeros([0, 4, 3])
        all_staff_prf1 = np.zeros([0, 3])
        line_thickness = (self.params.staff_line_found_distance - 1) // 2 + 1
        for single_data in data:
            pred_lines: StaffLines = StaffLines(single_data.pred.all_staff_lines())
            gt_lines: StaffLines = StaffLines(single_data.gt.all_staff_lines())

            pred_img = np.zeros(single_data.shape, dtype=np.int32)
            gt_img = np.zeros(single_data.shape, dtype=np.int32)
            pred_lines.draw(pred_img, 1, thickness=line_thickness)
            gt_lines.draw(gt_img, 1, thickness=line_thickness)

            # detect the closest lines

            def found_lines(from_lines: StaffLines, target_lines: StaffLines):
                target_label_img = np.zeros(single_data.shape, dtype=np.int32)
                for i, line in enumerate(target_lines):
                    line.draw(target_label_img, i + 1, thickness=line_thickness)

                target_img = (target_label_img > 0).astype(np.int32)

                hit_lines, single_lines = [], []
                for line in from_lines:
                    canvas = np.zeros(single_data.shape, dtype=np.int32)
                    line.draw(canvas, 3, thickness=line_thickness)
                    sum_img = canvas + target_img
                    if sum_img.max() == 4:
                        target_line_idx = (canvas * 1000 + target_label_img).max() - 3 * 1000 - 1
                        if self.params.debug:
                            print(target_line_idx)
                            f, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                            ax[0].imshow(canvas * 4 + target_label_img)
                            ax[1].imshow(target_img)
                            plt.show()
                        target_line = target_lines[target_line_idx]
                        target_canvas = np.zeros(canvas.shape, dtype=np.int32)
                        target_line.draw(target_canvas, 10, thickness=line_thickness)
                        canvas += target_canvas
                        total_line_hit = canvas.max(axis=0)
                        tp = (total_line_hit == 13).sum()
                        fp = (total_line_hit == 3).sum()
                        fn = (total_line_hit == 10).sum()
                        overlap = tp / (tp + fp + fn)

                        if overlap > self.params.line_hit_overlap_threshold:
                            hit_lines.append((line, target_line, precision_recall_f1(tp, fp, fn)))
                        else:
                            single_lines.append(line)


                    else:
                        single_lines.append(line)
                        if self.params.debug:
                            f, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                            ax[0].imshow(sum_img)
                            ax[1].imshow(target_img)
                            plt.show()

                return single_lines, hit_lines

            fn_lines = StaffLines(gt_lines[:])
            fp_lines, tp_line_pairs = found_lines(pred_lines, gt_lines)
            for i, (p_line, gt_line, _) in reversed(list(enumerate(tp_line_pairs))):
                if gt_line in fn_lines:
                    fn_lines.remove(gt_line)
                else:
                    logger.warning("Line hitted more than once")
                    del tp_line_pairs[i]
                    fp_lines.append(p_line)

            tp_pred_lines = [pred_line for pred_line, _, _ in tp_line_pairs]
            tp_lines = [gt_line for _, gt_line, _ in tp_line_pairs]

            logger.debug("TP: {}, FP: {}, FN: {} in file {}".format(len(tp_lines), len(fp_lines), len(fn_lines), single_data.label))

            # compute F1 for line detection
            line_found_counts = [len(tp_lines), len(fp_lines), len(fn_lines)]
            line_found_counts.append(np.sum(line_found_counts))

            # staves detection
            pred_staves: List[MusicLine] = single_data.pred[:]
            gt_staves: List[MusicLine] = single_data.gt[:]

            tp_staves = []
            fp_staves = []

            for pred_staff in pred_staves:
                pred_staff_lines = [l for l in pred_staff.staff_lines if l in tp_pred_lines]
                if len(pred_staff_lines) < self.params.staff_n_lines_threshold:
                    # not hit
                    fp_staves.append(pred_staff)
                    continue

                gt_staff_lines = [tp_lines[tp_pred_lines.index(l)] for l in pred_staff_lines]

                # test if all staff lines belong to the same gt staff
                r = list(set([gt_staff for gt_staff in gt_staves if any([gt_l in gt_staff.staff_lines for gt_l in gt_staff_lines])]))
                if len(r) == 1:
                    # hit
                    n_pred = len(pred_staff.staff_lines)
                    n_true = len(r[0].staff_lines)
                    n_share = len(pred_staff_lines)

                    tp = n_share
                    fp = n_pred - n_share
                    fn = n_true - n_share

                    tp_staves.append((pred_staff, r[0], precision_recall_f1(tp, fp, fn)))
                    gt_staves.remove(r[0])
                else:
                    # different staves
                    fp_staves.append(pred_staff)

            fn_staves = gt_staves
            staff_found_counts = [len(tp_staves), len(fp_staves), len(fn_staves)]
            staff_found_counts.append(np.sum(staff_found_counts))


            all_counts = np.concatenate((all_counts, [[line_found_counts, (0, 0, 0, 0), staff_found_counts, (0, 0, 0, 0)]]), axis=0)
            if len(tp_line_pairs) > 0:
                for _, _, prf1 in tp_line_pairs:
                    all_prf1 = np.concatenate((all_prf1, [[(0, 0, 0), prf1, (0, 0, 0), (0, 0, 0)]]), axis=0)

            if len(tp_staves) > 0:
                for _, _, prf1 in tp_staves:
                    all_staff_prf1 = np.concatenate((all_staff_prf1, [prf1]), axis=0)

        sum_counts = all_counts.sum(axis=0)
        mean_prf1 = all_prf1.mean(axis=0)
        mean_staff_prf1 = all_staff_prf1.mean(axis=0)
        mean_prf1[0] = precision_recall_f1(*tuple(sum_counts[0, :3]))
        mean_prf1[2] = precision_recall_f1(*tuple(sum_counts[2, :3]))
        mean_prf1[3] = mean_staff_prf1

        return sum_counts, mean_prf1


if __name__ == "__main__":
    from omr.stafflines.detection.pixelclassifier.predictor import BasicStaffLinePredictor
    from database import DatabaseBook
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

    book = DatabaseBook('Graduel')
    pcgts = PcGts.from_file(book.page('Graduel_de_leglise_de_Nevers_521').file('pcgts'))

    pred = BasicStaffLinePredictor(
        StaffLinePredictorParameters(
            None,
            StaffLineDetectionDatasetParams(
                full_page=False,
                gray=False,
            )
        )
    )
    predictions = []
    for p in pred.predict([pcgts]):
        predictions.append(
            EvaluationData(
                p.line.operation.page.location.local_path(),
                p.line.operation.music_lines,
                p.music_lines,
                p.line.operation.page_image.shape,
            )
        )

    evaluator = StaffLineDetectionEvaluator()
    evaluator.evaluate(predictions)
