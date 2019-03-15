from omr.stafflines.detection.predictor import StaffLinesPredictor, PredictorParameters
from database.file_formats.pcgts import PcGts, StaffLines, StaffLine, MusicLines
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


class StaffLineDetectionEvaluator:
    def __init__(self):
        self.line_hit_overlap_threshold = 0.1
        self.debug = False

    def evaluate(self, data: List[EvaluationData], staff_line_found_distance=5):
        all_counts = np.zeros([0, 2, 4])
        all_prf1 = np.zeros([0, 2, 3])
        for single_data in data:
            pred_lines: StaffLines = StaffLines(single_data.pred.all_staff_lines())
            gt_lines: StaffLines = StaffLines(single_data.gt.all_staff_lines())

            pred_img = np.zeros(single_data.shape, dtype=np.int32)
            gt_img = np.zeros(single_data.shape, dtype=np.int32)
            pred_lines.draw(pred_img, 1, thickness=staff_line_found_distance // 2)
            gt_lines.draw(gt_img, 1, thickness=staff_line_found_distance // 2)

            # detect the closest lines

            def found_lines(from_lines: StaffLines, target_lines: StaffLines):
                target_label_img = np.zeros(single_data.shape, dtype=np.int32)
                for i, line in enumerate(target_lines):
                    line.draw(target_label_img, i + 1, thickness=staff_line_found_distance // 2)

                target_img = (target_label_img > 0).astype(np.int32)

                hit_lines, single_lines = [], []
                for line in from_lines:
                    canvas = np.zeros(single_data.shape, dtype=np.int32)
                    line.draw(canvas, 3, thickness=staff_line_found_distance // 2)
                    sum_img = canvas + target_img
                    if sum_img.max() == 4:
                        target_line_idx = (canvas * 1000 + target_label_img).max() - 3 * 1000 - 1
                        if self.debug:
                            print(target_line_idx)
                            f, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                            ax[0].imshow(canvas * 4 + target_label_img)
                            ax[1].imshow(target_img)
                            plt.show()
                        target_line = target_lines[target_line_idx]
                        target_canvas = np.zeros(canvas.shape, dtype=np.int32)
                        target_line.draw(target_canvas, 10, thickness=staff_line_found_distance // 2)
                        canvas += target_canvas
                        total_line_hit = canvas.max(axis=0)
                        tp = (total_line_hit == 13).sum()
                        fp = (total_line_hit == 3).sum()
                        fn = (total_line_hit == 10).sum()
                        overlap = tp / (tp + fp + fn)

                        if overlap > self.line_hit_overlap_threshold:
                            hit_lines.append((line, target_line, precision_recall_f1(tp, fp, fn)))
                        else:
                            single_lines.append(line)


                    else:
                        single_lines.append(line)
                        if self.debug:
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

            tp_lines = [gt_line for _, gt_line, _ in tp_line_pairs]

            logger.debug("TP: {}, FP: {}, FN: {} in file {}".format(len(tp_lines), len(fp_lines), len(fn_lines), single_data.label))

            # compute F1 for line detection
            line_found_counts = [len(tp_lines), len(fp_lines), len(fn_lines)]
            line_found_counts.append(np.sum(line_found_counts))

            all_counts = np.concatenate((all_counts, [[line_found_counts, (0, 0, 0, 0)]]), axis=0)
            if len(tp_line_pairs) > 0:
                for _, _, prf1 in tp_line_pairs:
                    all_prf1 = np.concatenate((all_prf1, [[(0, 0, 0), prf1]]), axis=0)

        sum_counts = all_counts.sum(axis=0)
        mean_prf1 = all_prf1.mean(axis=0)
        mean_prf1[0] = precision_recall_f1(*tuple(sum_counts[0,:3]))

        return sum_counts, mean_prf1


if __name__ == "__main__":
    from omr.stafflines.detection.pixelclassifier.predictor import BasicStaffLinePredictor
    from database import DatabaseBook

    book = DatabaseBook('Graduel')
    pcgts = PcGts.from_file(book.page('Graduel_de_leglise_de_Nevers_521').file('pcgts'))

    pred = BasicStaffLinePredictor(
        PredictorParameters(
            None,
            full_page=False,
            gray=False,
        )
    )
    evaluator = StaffLineDetectionEvaluator(pred)
    evaluator.evaluate([pcgts])
