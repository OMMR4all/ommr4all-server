from database.file_formats.pcgts import StaffLines, Line, Point, Coords, Size, Rect
from typing import List, NamedTuple, Tuple, Union, Iterable
import numpy as np

import logging

from omr.experimenter.experimenter import EvaluatorParams
from omr.steps.symboldetection.evaluator import precision_recall_f1

logger = logging.getLogger(__name__)


class EvaluationData(NamedTuple):
    label: str
    gt: List[Line]
    pred: List[Line]
    shape: Tuple[int, int]

    @staticmethod
    def from_json(d: dict):
        return EvaluationData(d['label'],
                              [Line.from_json(l) for l in d['gt']],
                              [Line.from_json(l) for l in d['pred']],
                              tuple(d['shape']))

    def to_json(self):
        return {
            'label': self.label,
            'gt': [l.to_json() for l in self.gt],
            'pred': [l.to_json() for l in self.pred],
            'shape': list(self.shape)
        }

    def _scale(self, p: Union[Coords, Point, float, int], scale: float):
        if isinstance(p, Coords):
            return p.scale(scale)
        elif isinstance(p, Point):
            return p.scale(scale)
        elif isinstance(p, Size):
            return p.scale(scale)
        elif isinstance(p, Rect):
            return Rect(self._scale(p.origin, scale), self._scale(p.size, scale))
        elif isinstance(p, Iterable):
            return np.array(p) * scale
        else:
            return p * scale

    def page_to_eval_scale(self, p: Union[Coords, Point, float, int]):
        return self._scale(p, self.shape[0])


class StaffLineDetectionEvaluator:
    def __init__(self, params=None):
        self.params = params if params else EvaluatorParams()

    def evaluate(self, data: List[EvaluationData]) \
            -> Tuple[np.ndarray, np.ndarray,
                     Tuple[
                         List[List[Tuple[Line, Line, np.ndarray]]],
                         List[List[Line]],
                         List[List[Line]]
                         ]]:
        def all_staff_lines(line: List[Line]):
            return sum([l.staff_lines for l in line], [])

        all_counts = np.zeros([0, 4, 4])
        all_prf1 = np.zeros([0, 4, 3])
        all_staff_prf1 = np.zeros([0, 3])
        all_tp_staves: List[Tuple[Line, Line, np.ndarray]] = []
        all_fp_staves: List[List[Line]] = []
        all_fn_staves: List[List[Line]] = []
        line_thickness = (self.params.staff_line_found_distance - 1) // 2 + 1
        for single_data in data:
            pred_lines: StaffLines = StaffLines(all_staff_lines(single_data.pred))
            gt_lines: StaffLines = StaffLines(all_staff_lines(single_data.gt))

            pred_img = np.zeros(single_data.shape, dtype=np.int32)
            gt_img = np.zeros(single_data.shape, dtype=np.int32)
            for l in pred_lines:
                single_data.page_to_eval_scale(l.coords).draw(pred_img, 1, thickness=line_thickness)
            for l in gt_lines:
                single_data.page_to_eval_scale(l.coords).draw(gt_img, 1, thickness=line_thickness)


            # detect the closest lines
            def found_lines(from_lines: StaffLines, target_lines: StaffLines):
                target_label_img = np.zeros(single_data.shape, dtype=np.int32)
                for i, line in enumerate(target_lines):
                    single_data.page_to_eval_scale(line.coords).draw(target_label_img, i + 1, thickness=line_thickness)

                target_img = (target_label_img > 0).astype(np.int32)

                hit_lines, single_lines = [], []
                for line in from_lines:
                    canvas = np.zeros(single_data.shape, dtype=np.int32)
                    single_data.page_to_eval_scale(line.coords).draw(canvas, 3, thickness=line_thickness)
                    sum_img = canvas + target_img
                    if sum_img.max() == 4:
                        target_line_idx = (canvas * 1000 + target_label_img).max() - 3 * 1000 - 1
                        if self.params.debug:
                            import matplotlib.pyplot as plt
                            print(target_line_idx)
                            f, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                            ax[0].imshow(canvas * 4 + target_label_img)
                            ax[1].imshow(target_img)
                            plt.show()
                        target_line = target_lines[target_line_idx]
                        target_canvas = np.zeros(canvas.shape, dtype=np.int32)
                        single_data.page_to_eval_scale(target_line.coords).draw(target_canvas, 10, thickness=line_thickness)
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
                            import matplotlib.pyplot as plt
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
            pred_staves: List[Line] = single_data.pred[:]
            gt_staves: List[Line] = single_data.gt[:]

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

            all_tp_staves.append(tp_staves)
            all_fp_staves.append(fp_staves)
            all_fn_staves.append(fn_staves)

        sum_counts = all_counts.sum(axis=0)
        mean_prf1 = all_prf1.mean(axis=0)
        mean_staff_prf1 = all_staff_prf1.mean(axis=0)
        mean_prf1[0] = precision_recall_f1(*tuple(sum_counts[0, :3]))
        mean_prf1[2] = precision_recall_f1(*tuple(sum_counts[2, :3]))
        mean_prf1[3] = mean_staff_prf1

        return sum_counts, mean_prf1, (all_tp_staves, all_fp_staves, all_fn_staves)
