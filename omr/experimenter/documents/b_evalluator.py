from dataclasses import dataclass
from typing import List
import logging

import numpy as np

from database.file_formats.pcgts import MusicSymbol, Line
from database.file_formats.pcgts.page import Connection, SyllableConnector
from omr.dataset import DatasetParams
from omr.experimenter.experimenter import SingleDataArgs, GlobalDataArgs, EvaluatorParams
from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, SymbolMelodyEvaluator, precision_recall_f1, \
    Counts
from omr.steps.text.experimenter import TextExperimenter

logger = logging.getLogger(__name__)


def evaluate_symbols(pred_symbols: List[List[MusicSymbol]], gt_symbols: List[List[MusicSymbol]]):
    evaluator = SymbolDetectionEvaluator()
    metrics, counts, acc_counts, acc_acc, total_diffs, total_diffs_count = evaluator.evaluate(gt_symbols, pred_symbols)
    melody_evaluator = SymbolMelodyEvaluator()
    prec_rec_f1 = np.array([precision_recall_f1(c[Counts.TP], c[Counts.FP], c[Counts.FN]) for c in counts])

    melody_counts, melody_acc = melody_evaluator.evaluate(gt_symbols, pred_symbols)

    results = [[prec_rec_f1, acc_acc, total_diffs, 0, 0, melody_acc, total_diffs_count]]

    prec_rec_f1_list = [r[0] for r in results if r is not None]
    acc_counts_list = [r[1] for r in results if r is not None]
    total_diffs = [r[2] for r in results if r is not None]
    # count_1 = [r[3] for r in results if r is not None]
    # clef_analysis = [r[4] for r in results if r is not None]
    melody_count_list = [r[5] for r in results if r is not None]
    # total_diffs_count_list = [r[6] for r in results if r is not None]

    if len(prec_rec_f1_list) == 0:
        return

    prf1_mean = np.mean(prec_rec_f1_list, axis=0)
    prf1_std = np.std(prec_rec_f1_list, axis=0)

    acc_mean = np.mean(acc_counts_list, axis=0)
    acc_std = np.std(acc_counts_list, axis=0)
    diffs_mean = np.mean(total_diffs, axis=0)
    diffs_std = np.std(total_diffs, axis=0)
    # count1_mean = np.mean(count_1, axis=0)
    # count1_std = np.std(count_1, axis=0)
    # clef_analysis_sum = np.sum(clef_analysis, axis=0)
    # clef_analysis_mean = np.mean(clef_analysis, axis=0)
    # clef_analysis_std = np.std(clef_analysis, axis=0)

    melody_mean = np.mean(melody_count_list, axis=0)
    melody_std = np.std(melody_count_list, axis=0)
    # total_diffs_sum = np.sum(total_diffs_count_list, axis=0)
    all_symbol_detection = np.array(
        sum([[prf1_mean[1:, i], prf1_std[1:, i]] for i in range(3)], [])).transpose().reshape(-1)
    all_acc = np.array(np.transpose([acc_mean[:, 0], acc_std[:, 0]]).reshape([-1]))
    all_diffs = np.array(np.transpose([diffs_mean, diffs_std])).reshape([-1])
    # all_count1 = np.array(np.transpose([count1_mean, count1_std])).reshape([-1])
    # all_clef_analysis = np.array(np.transpose([clef_analysis_sum, clef_analysis_mean])).reshape([-1])
    all_melody = np.array(np.transpose([melody_mean, melody_std])).reshape([-1])
    # total_diffs_sum = np.array(total_diffs_sum)
    output_String = "{}{}".format("EXPERIMENT_OUT=",
                                  ','.join(map(str, list(all_symbol_detection) + list(all_acc) + list(
                                      all_diffs) + list(all_melody))))  # + list(total_diffs_sum))))

    output_String = output_String[len("EXPERIMENT_OUT="):]
    output_array = output_String.split(",")
    excel_lines = []
    header = "All Symbols"
    excel_lines.append(
        ["All Symbols"] + ["~"] * 5 + ["Notes"] + ["~"] * 5 + ["Clefs"] + ["~"] * 5 + ["Accidentals"] + ["~"] * 5 + [
            "Note GC"] + ["~"] + ["Note PIS"] + ["~"] + ["Clef type"] + ["~"] +
        ["Clef pis"] + ["~"] + ["Accid type"] + ["~"] + ["hsar"] + ["~"] + ["dsar"] + ["~"] + ["neume seq"] + ["~"] +
        ["missing notes"] + ["~"] + ["Wrong Nc"] + ["~"] + ["Wrong PIS"] + ["~"] + ["Missing Clefs"] + ["~"] + [
            "Missing Accids"] + ["~"]
        + ["Add notes"] + ["~"] + ["FP Wrong NC"] + ["~"] + ["FP Wrong Pis"] + ["~"]
        + ["Add Clefs"] + ["~"] + ["Add Accids"] + ["~"] + ["Acc"] + ["~"] + ["Total"] + ["~"] + ["Melody"] + ["~"])
    excel_lines.append(["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Acc"] + ["~"] + ["Acc"] + ["~"] + ["Acc"] + ["~"] + ["Acc"] + ["~"] +
                       ["Accid Type"] + ["~"] + ["CAR"] + ["~"] + ["CAR"] + ["~"] + ["NAR"] + ["~"] +
                       ["Amount", "~"] * 13
                       )
    excel_lines.append(output_array)

    return excel_lines


def evaluate_text(pred_text: List[str], gt_text: List[str]):
    ##pred_text, gt_text = prepare_text_gt(pred_book, gt_book)
    s_data = SingleDataArgs(0, None, None, None, None,
                            GlobalDataArgs("EXPERIMENT_OUT=", None, None, None, None, None, None, None, DatasetParams(),
                                           None, None, None, None, None,
                                           None, None, None, None))
    exp = TextExperimenter(s_data, logger)
    results = exp.evaluate((pred_text, gt_text), EvaluatorParams(debug=False))
    # results = counts, prf1, (all_tp_staves, all_fp_staves, all_fn_staves)
    # print(results)
    output_string = exp.print_results(
        GlobalDataArgs("EXPERIMENT_OUT=", None, None, None, None, None, None, None, None, None, None, None, None, None,
                       None, None, None, None), [results], logger)
    output_array = output_string[len("EXPERIMENT_OUT="):].split(",")
    excel_lines = []

    excel_lines.append(
        ['#', "~", 'avg_ler', "~", '#chars', "~", '#errs', "~", '#sync_errs', "~", '#sylls', "~", 'avg_ser', "~",
         '#words', "~", "avg_wer", "~", "#conf", "~", 'conf_err', "~"])
    excel_lines.append(output_array)
    return excel_lines

@dataclass
class SyllableEvalInput:
    pred_annotation: List[Connection]
    gt_annotation: List[Connection]
    p_line: Line
    gt_line: Line

def evaluate_syllabels(eval_data: List[SyllableEvalInput]):
    def get_all_connections_of_music_line(line: Line, connections: List[Connection]) -> List[SyllableConnector]:
        syl_connectors: List[SyllableConnector] = []
        for i in connections:
            for t in i.text_region.lines:
                if t.id == line.id:  # Todo
                    syl_connectors += i.syllable_connections
                pass
        return syl_connectors
    def prepare_syllable_gt(eval_data: List[SyllableEvalInput]):



        tp = 0
        fp = 0
        fn = 0
        for i in eval_data:
            pred_annotations = i.pred_annotation
            gt_annotations = i.gt_annotation
            p_line = i.p_line
            gt_line = i.gt_line

            syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
            syl_connectors_gt = get_all_connections_of_music_line(gt_line, gt_annotations)
            for i in syl_connectors_gt:
                found = False
                index = -1

                for ind, p in enumerate(syl_connectors_pred):
                    if i.note.coord.x == p.note.coord.x:
                        if i.syllable.text == p.syllable.text:
                            tp += 1
                            found = True
                            del syl_connectors_pred[ind]
                            break
                        else:
                            print(f'gt {i.syllable.text} p: {p.syllable.text}')
                if not found:
                    fn += 1
            fp += len(syl_connectors_pred)
        #print(tp)
        #print(fp)
        #print(fn)
        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
        acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1
        #print(f1)
        #print(acc)
        return tp, fn, fp, f1, acc

    def prepare_syllable_gt_continuation_error(eval_data: List[SyllableEvalInput]):

        tp = 0
        fp = 0
        fn = 0
        cn = 0
        for i in eval_data:
            pred_annotations = i.pred_annotation
            gt_annotations = i.gt_annotation
            p_line = i.p_line
            gt_line = i.gt_line

            syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
            syl_connectors_gt: List[SyllableConnector] = get_all_connections_of_music_line(gt_line, gt_annotations)
            consecutive = False
            for i in sorted(syl_connectors_gt, key=lambda x: x.note.coord.x):
                found = False
                index = -1
                for ind, p in enumerate(syl_connectors_pred):
                    if i.note.coord.x == p.note.coord.x:
                        if i.syllable.text == p.syllable.text:
                            consecutive = False
                            tp += 1
                            found = True
                            del syl_connectors_pred[ind]
                            break
                        else:
                            print(f'gt {i.syllable.text} p: {p.syllable.text}')
                if not found:
                    if consecutive:
                        cn+= 1
                    fn += 1
                    consecutive = True
            fp += len(syl_connectors_pred)
        print(tp)
        print(fp)
        print(fn)
        tp = tp + cn
        fn = fn - cn
        fp = fp - cn
        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
        acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1
        print(f1)
        print(acc)
        return tp, fn, fp, f1, acc

    result = prepare_syllable_gt(eval_data)
    result2 = prepare_syllable_gt_continuation_error(eval_data)
    excel_lines = []
    res2_labels = ["cTp", "cFn", "cFP", "cF1", "c_acc"]
    excel_lines.append(["Tp", "Fn", "FP", "F1", "acc"] + res2_labels)

    excel_lines.append(result + result2)
    return excel_lines
