import os

import cv2
from PIL import Image

from database.file_formats.pcgts import PageScaleReference, MusicSymbol, Point
from omr.experimenter.experimenter import Experimenter, EvaluatorParams
from database import DatabaseBook, DatabasePage
from omr.steps.symboldetection.dataset import PcGts
from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, Counts, precision_recall_f1, AccCounts, \
    SymbolErrorTypeDetectionEvaluator, SymbolMelodyEvaluator
from typing import List, Tuple
from prettytable import PrettyTable
import numpy as np


class SymbolsExperimenter(Experimenter):
    def output_debug_images(self, predictions, params=None):
        debug_path = os.path.join(self.args.model_dir, 'debug')
        if not os.path.exists(debug_path):
            os.mkdir(debug_path)
        params = params if params else EvaluatorParams()
        min_distance_sqr = params.symbol_detected_min_distance ** 2
        def extract_coords_of_symbols(symbols: List[MusicSymbol]) -> List[Tuple[Point, MusicSymbol]]:
            return [(s.coord, s) for s in symbols]
        for page in predictions:
            page_dataset_page: DatabasePage = page.dataset_page
            page_pcgts: PcGts = page.pcgts
            scale_reference = PageScaleReference.NORMALIZED_X2
            page_ = page_pcgts.page
            file = 'color'
            img = np.array(Image.open(page_.location.file(scale_reference.file(file)).local_path()))
            overlay = img.copy()
            def scale(x):
                return np.round(page_.page_to_image_scale(x, scale_reference)).astype(int)
            avg_line_distance = page_.page_to_image_scale(page_.avg_staff_line_distance(), scale_reference)
            for p in page.music_lines:
                #p_symbols = extract_coords_of_symbols(pred)
                #gt_symbols_orig = extract_coords_of_symbols(gt)
                #gt_symbols = gt_symbols_orig[:]
                p_symbols = extract_coords_of_symbols(p.symbols)
                gt_symbols = extract_coords_of_symbols(p.line.operation.music_line.symbols)

                pairs =[]

                for p_i, (p_c, p_s) in reversed(list(enumerate(p_symbols))):
                    best_d = 10000
                    best_s = None
                    best_gti = None
                    # closest other symbol

                    for gt_i, (gt_c, gt_s) in enumerate(gt_symbols):
                        d = gt_c.distance_sqr(p_c)

                        if d > min_distance_sqr:
                            continue

                        if d < best_d:
                            best_s = (gt_c, gt_s)
                            best_d = d
                            best_gti = gt_i

                    if best_s:
                        pairs.append(((p_c, p_s), best_s))
                        del gt_symbols[best_gti]
                        del p_symbols[p_i]

                def color_picker(symbol1: MusicSymbol, symbol2: MusicSymbol):
                    if symbol1 is None:
                        return (255, 0, 255)
                    elif symbol2 is None:
                        return (255, 0, 0)
                    elif symbol1.symbol_type != symbol2.symbol_type:
                        return (0, 255, 237)

                    elif symbol1.position_in_staff != symbol2.position_in_staff:
                        if symbol1.symbol_type == symbol1.symbol_type.CLEF:
                            print("{}, {}".format(symbol1.position_in_staff, symbol2.position_in_staff))
                        return (255, 237, 0)
                    elif symbol1.graphical_connection != symbol2.graphical_connection:
                        return (0, 255, 0)
                    return (0, 0, 0)
                for predict, gt in pairs:
                    symbol1 = predict[1]
                    symbol2 = gt[1]

                    pos = tuple(scale(predict[1].coord.p))
                    #cv2.circle(img, pos, int(avg_line_distance // 3), color=color_picker(symbol1, symbol2), thickness=-1)
                    cv2.rectangle(img, (pos[0] + int(avg_line_distance // 3), pos[1] + int(avg_line_distance // 3)) ,
                                  (pos[0] - int(avg_line_distance // 3), pos[1] - int(avg_line_distance // 3)), color=color_picker(symbol1, symbol2), thickness=2)
                for symbol in p_symbols:
                    pos = tuple(scale(symbol[1].coord.p))
                    #cv2.circle(img, pos, int(avg_line_distance // 3), color=color_picker(symbol[1], None), thickness=-1)
                    cv2.rectangle(img, (pos[0] + int(avg_line_distance // 3), pos[1] + int(avg_line_distance // 3)),
                                  (pos[0] - int(avg_line_distance // 3), pos[1] - int(avg_line_distance // 3)),
                                  color=color_picker(symbol[1], None), thickness=2)
                for symbol in gt_symbols:
                    pos = tuple(scale(symbol[1].coord.p))
                    #cv2.circle(img, pos, int(avg_line_distance // 3), color=color_picker(None, symbol[1]), thickness=-1)
                    cv2.rectangle(img, (pos[0] + int(avg_line_distance // 3), pos[1] + int(avg_line_distance // 3)),
                                  (pos[0] - int(avg_line_distance // 3), pos[1] - int(avg_line_distance // 3)),
                                  color=color_picker(None, symbol[1]), thickness=2)
            alpha = 0.4
            img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)
            cv2.putText(img, "TP", (scale(0.03), scale(0.03)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(0, 0, 0), thickness=2)
            cv2.putText(img, "FP", (scale(0.03), scale(0.06)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(255, 0, 0), thickness=2)
            cv2.putText(img, "FN", (scale(0.03), scale(0.09)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(255, 0, 255), thickness=2)
            cv2.putText(img, "PiS", (scale(0.03), scale(0.12)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(255, 237, 0), thickness=2)
            cv2.putText(img, "Con", (scale(0.03), scale(0.15)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(0, 255, 0), thickness=2)
            cv2.putText(img, "Type", (scale(0.03), scale(0.18)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(0, 255, 237), thickness=2)

            Image.fromarray(img).save(os.path.join(debug_path, page_pcgts.page.location.page + ".png"))

    def extract_gt_prediction(self, full_predictions):
        output = zip(*[(p.line.operation.music_line.symbols, p.symbols) for p in sum([p.music_lines for p in full_predictions], [])]), full_predictions
        return output

    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
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
            for ml in p.music_lines:
                o_pcgts = output_pcgts_by_page_name[ml.line.operation.page.location.page]
                o_pcgts.page.music_line_by_id(ml.line.operation.music_line.id).symbols = ml.symbols

    def evaluate(self, predictions, evaluation_params):
        symbols, full_predictions = predictions
        gt_symbols, pred_symbols = symbols
        evaluator = SymbolDetectionEvaluator(evaluation_params)
        metrics, counts, acc_counts, acc_acc, total_diffs, total_diffs_count = evaluator.evaluate(gt_symbols, pred_symbols)
        evaluator2 = SymbolErrorTypeDetectionEvaluator(evaluation_params)
        counts1, counts2 = evaluator2.evaluate(full_predictions)
        melody_evaluator = SymbolMelodyEvaluator(evaluation_params)
        melody_counts, melody_acc = melody_evaluator.evaluate(gt_symbols, pred_symbols)

        at = PrettyTable()

        at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])
        at.add_column("TP", counts[:, Counts.TP])
        at.add_column("FP", counts[:, Counts.FP])
        at.add_column("FN", counts[:, Counts.FN])

        prec_rec_f1 = np.array([precision_recall_f1(c[Counts.TP], c[Counts.FP], c[Counts.FN]) for c in counts])
        self.fold_log.debug(prec_rec_f1)

        at.add_column("Precision", prec_rec_f1[:, 0])
        at.add_column("Recall", prec_rec_f1[:, 1])
        at.add_column("F1", prec_rec_f1[:, 2])
        self.fold_log.debug(at.get_string())

        at = PrettyTable()
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC", "Neume Sequence"])
        at.add_column("True", acc_counts[:, AccCounts.TRUE])
        at.add_column("False", acc_counts[:, AccCounts.FALSE])
        at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
        at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

        self.fold_log.debug(at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
        at.add_row(total_diffs)
        self.fold_log.debug(at)

        return prec_rec_f1, acc_acc, total_diffs, counts1.to_np_array(), counts2.to_np_array(), melody_acc, total_diffs_count

    @classmethod
    def print_results(cls, args, results, log):
        log.info("Total Result:")

        prec_rec_f1_list = [r[0] for r in results if r is not None]
        acc_counts_list = [r[1] for r in results if r is not None]
        total_diffs = [r[2] for r in results if r is not None]
        count_1 = [r[3] for r in results if r is not None]
        clef_analysis = [r[4] for r in results if r is not None]
        melody_count_list = [r[5] for r in results if r is not None]
        total_diffs_count_list = [r[6] for r in results if r is not None]

        if len(prec_rec_f1_list) == 0:
            return

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        acc_mean = np.mean(acc_counts_list, axis=0)
        acc_std = np.std(acc_counts_list, axis=0)
        diffs_mean = np.mean(total_diffs, axis=0)
        diffs_std = np.std(total_diffs, axis=0)
        count1_mean = np.mean(count_1, axis=0)
        count1_std = np.std(count_1, axis=0)
        clef_analysis_sum = np.sum(clef_analysis, axis=0)
        clef_analysis_mean = np.mean(clef_analysis, axis=0)
        clef_analysis_std = np.std(clef_analysis, axis=0)

        melody_mean = np.mean(melody_count_list, axis=0)
        melody_std = np.std(melody_count_list, axis=0)
        total_diffs_sum = np.sum(total_diffs_count_list, axis=0)

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
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC", "Neume Sequence"])
        at.add_column("Accuracy [%]", acc_mean[:, 0] * 100)
        at.add_column("+- [%]", acc_std[:, 0] * 100)

        log.info("\n" + at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
        at.add_row(diffs_mean)
        at.add_row(diffs_std)
        log.info("\n" + at.get_string())

        if args.magic_prefix:
            all_symbol_detection = np.array(sum([[prf1_mean[1:, i], prf1_std[1:, i]] for i in range(3)], [])).transpose().reshape(-1)
            all_acc = np.array(np.transpose([acc_mean[:, 0], acc_std[:, 0]]).reshape([-1]))
            all_diffs = np.array(np.transpose([diffs_mean, diffs_std])).reshape([-1])
            all_count1 = np.array(np.transpose([count1_mean, count1_std])).reshape([-1])
            all_clef_analysis = np.array(np.transpose([clef_analysis_sum, clef_analysis_mean])).reshape([-1])
            all_melody = np.array(np.transpose([melody_mean, melody_std])).reshape([-1])
            total_diffs_sum = np.array(total_diffs_sum)
            print("{}{}".format(args.magic_prefix, ','.join(map(str, list(all_symbol_detection) + list(all_acc) + list(all_diffs) + list(all_melody) +  list(all_clef_analysis) + list(total_diffs_sum)))))
