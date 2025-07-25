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


def evaluate_symbols(pred_symbols: List[List[MusicSymbol]], gt_symbols: List[List[MusicSymbol]], count_output=False):
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


    ##counts
    counts_all = np.array([[c[Counts.TP], c[Counts.FP], c[Counts.FN]] for c in counts]).reshape([-1])
    counts_acc = np.array([[c[Counts.TP], c[Counts.FP], c[Counts.FN]] for c in acc_counts]).reshape([-1])

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
                                      all_diffs) + list(all_melody) + list(counts_all) +list(counts_acc) )))   # + list(total_diffs_sum))))

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
        + ["Add Clefs"] + ["~"] + ["Add Accids"] + ["~"] + ["Acc"] + ["~"] + ["Total"] + ["~"] + ["Melody"] + ["~"] + ["Existence"]+["Existence"]+["Existence"]
        +["TypeAll"]+["TypeAll"]+["TypeAll"] +["TypeNotes", "TypeNotes", "TypeNotes"]+["TypeClef", "TypeClef", "TypeClef"]
        +["TypeAccid", "TypeAccid", "TypeAccid"] + ["NoteSubtype", "NoteSubtype", "NoteSubtype"] + ["NotePis", "NotePis", "NotePis"] +
        ["ClefSubtype", "ClefSubtype", "ClefSubtype"] + ["ClefPis", "ClefPis", "ClefPis"] +["AccidSubtype", "AccidSubtype", "AccidSubtype"] +
        ["Sequence", "Sequence", "Sequence"] + ["SequenceNC", "SequenceNC", "SequenceNC"] + ["SequenceNeume", "SequenceNeume", "SequenceNeume"]


    )
    excel_lines.append(["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Precision"] + ["~"] + ["Recall"] + ["~"] + ["F1"] + ["~"] +
                       ["Acc"] + ["~"] + ["Acc"] + ["~"] + ["Acc"] + ["~"] + ["Acc"] + ["~"] +
                       ["Accid Type"] + ["~"] + ["CAR"] + ["~"] + ["CAR"] + ["~"] + ["NAR"] + ["~"] +
                       ["Amount", "~"] * 13 + ["#TP", "#FP", "#FN"]*5+ ["#True", "#False", "#Total"]*8
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
    gt_page: str
    pred_page: str


def evaluate_syllabels(eval_data: List[SyllableEvalInput], save_to_excel: bool = False, gtpage = "", predpage = ""):
    def get_all_connections_of_music_line(line: Line, connections: List[Connection]) -> List[SyllableConnector]:
        syl_connectors: List[SyllableConnector] = []
        for i in connections:
            for t in i.text_region.lines:
                if t.id == line.id:  # Todo
                    syl_connectors += i.syllable_connections
                pass
        return syl_connectors

    def prepare_syllable_gt(eval_data: List[SyllableEvalInput]):
        errors = []
        @dataclass()
        class SyllableErrorType:
            NoteError: int = 0
            TextError: int = 0
            Delete: int = 0
            Insert: int = 0
            PreviousLine: int = 0

        error_type = SyllableErrorType()
        tp = 0
        fp = 0
        fn = 0
        tp_withouttxt = 0
        fp_withouttxt = 0
        fn_withouttxt = 0

        fp_prev_line = []
        for i in eval_data:
            pred_annotations = i.pred_annotation
            gt_annotations = i.gt_annotation
            p_line = i.p_line
            gt_line = i.gt_line

            syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
            syl_connectors_gt = get_all_connections_of_music_line(gt_line, gt_annotations)
            add_gt_connections = []

            for i in syl_connectors_gt:
                found = False
                found_without_txt = False
                index = -1

                for ind, p in enumerate(syl_connectors_pred):
                    if abs(i.note.coord.x - p.note.coord.x) < 0.005:
                        tp_withouttxt += 1
                        found_without_txt = True
                        # (i.syllable.text in p.syllable.text or p.syllable.text in i.syllable.text) when different grammar used to split words in syllabels
                        if i.syllable.text.lower() == p.syllable.text.lower() or (i.syllable.text.lower() in p.syllable.text.lower() or p.syllable.text.lower() in i.syllable.text.lower()):
                            tp += 1
                            found = True
                            del syl_connectors_pred[ind]
                            break
                        else:
                            pass
                            #print(f'gt {i.syllable.text} p: {p.syllable.text}')
                if not found:
                    add_gt_connections.append(i)
                    fn += 1
                if not found_without_txt:
                    fn_withouttxt += 1
            fp += len(syl_connectors_pred)
            keep= True
            #print(len(add_gt_connections))
            once = True
            while keep:
                #print(len(add_gt_connections))
                keep = False
                if once and len(fp_prev_line) > 0 and len(add_gt_connections) > 0 and add_gt_connections[0].syllable.text.lower() == fp_prev_line[-1].syllable.text.lower():
                    errors.append(("PreviousLine", add_gt_connections[0].syllable.text.lower(), fp_prev_line[-1].syllable.text.lower()))

                    del add_gt_connections[0]
                    error_type.PreviousLine += 1
                    error_type.Delete -= 1
                    pass
                once = False
                for ind1, i in enumerate(add_gt_connections):
                    found = False

                    for ind2, p in enumerate(syl_connectors_pred):

                        if abs(i.note.coord.x - p.note.coord.x) < 0.005:
                            error_type.TextError += 1
                            del syl_connectors_pred[ind2]
                            del add_gt_connections[ind1]
                            errors.append(("Texterror", i.syllable.text.lower(),
                                           p.syllable.text.lower()))

                            found = True
                            break
                        if i.syllable.text.lower() == p.syllable.text.lower():
                            error_type.NoteError += 1
                            del syl_connectors_pred[ind2]
                            del add_gt_connections[ind1]
                            found = True
                            errors.append(("Noteerror", i.syllable.text.lower(),
                                           p.syllable.text.lower()))
                            break

                    if found:
                        keep = True
                        break
            for f in add_gt_connections:
                errors.append(("Insert", f.syllable.text.lower(), "None"))
            for g in syl_connectors_pred:
                errors.append(("Delete", "None", g.syllable.text.lower()))
            error_type.Insert += len(add_gt_connections)
            error_type.Delete += len(syl_connectors_pred)
            fp_prev_line = syl_connectors_pred

        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
        acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1
        # print(f1)
        # print(acc)

        return (tp, fn, fp, f1, acc, error_type.NoteError, error_type.TextError, \
            error_type.Insert, error_type.Delete, error_type.PreviousLine), errors

    def prepare_syllable_gt_continuation_error(eval_data: List[SyllableEvalInput]):

        tp = 0
        fp = 0
        fn = 0
        for i in eval_data:
            pred_annotations = i.pred_annotation
            gt_annotations = i.gt_annotation
            p_line = i.p_line
            gt_line = i.gt_line

            syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
            syl_connectors_gt: List[SyllableConnector] = get_all_connections_of_music_line(gt_line, gt_annotations)
            consecutive = False
            cn = 0

            for i in sorted(syl_connectors_gt, key=lambda x: x.note.coord.x):
                found = False
                index = -1
                for ind, p in enumerate(syl_connectors_pred):
                    if abs(i.note.coord.x - p.note.coord.x) < 0.005:
                        if i.syllable.text == p.syllable.text:
                            consecutive = False
                            tp += 1
                            found = True
                            del syl_connectors_pred[ind]
                            break
                        else:
                            pass
                            #print(f'gt {i.syllable.text} p: {p.syllable.text}')
                if not found:
                    if consecutive:
                        cn += 1
                        tp += 1
                    else:
                        fn += 1
                        consecutive = True
            fp += len(syl_connectors_pred)
        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
        acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1

        return tp, fn, fp, f1, acc

    def evaluate_melody_blocks(eval_data: List[SyllableEvalInput]):
        tp = 0
        fp = 0
        fn = 0
        for i in eval_data:
            pred_annotations = i.pred_annotation
            gt_annotations = i.gt_annotation
            p_line = i.p_line
            gt_line = i.gt_line

            syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
            syl_connectors_gt: List[SyllableConnector] = get_all_connections_of_music_line(gt_line, gt_annotations)
            syl_connectors_pred = list(set([i.note for i in syl_connectors_pred if len(i.syllable.text) > 0]))
            syl_connectors_gt = list(set([i.note for i in syl_connectors_gt if len(i.syllable.text) > 0]))
            for t in sorted(syl_connectors_gt, key=lambda x: x.coord.x):
                found = False
                index = -1
                for ind, p in enumerate(syl_connectors_pred):
                    if abs(t.coord.x - p.coord.x) < 0.005:
                        tp += 1
                        found = True
                        del syl_connectors_pred[ind]
                        break
                if not found:
                    fn += 1
            fp += len(syl_connectors_pred)

        p = tp / (tp + fp) if (tp + fp) > 0 else 1
        r = tp / (tp + fn) if (tp + fn) > 0 else 1
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
        acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1

        return tp, fn, fp, f1, acc
        pass

    result, errors = prepare_syllable_gt(eval_data)
    result2 = prepare_syllable_gt_continuation_error(eval_data)
    result3 = evaluate_melody_blocks(eval_data)
    excel_lines = []
    res2_labels = ["cTp", "cFn", "cFP", "cF1", "c_acc"]
    res3_labels = ["bTp", "bFn", "bFP", "bF1", "b_acc"]

    excel_lines.append(["Tp", "Fn", "FP", "F1", "acc", "NoteError", "TextError", "Insert", "Delete", "PrevLineError"] + res2_labels + res3_labels)
    excel_lines.append(result + result2 + result3)
    return excel_lines, errors
