from typing import List, NamedTuple

import numpy as np

from database.file_formats import PcGts
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Connection, SyllableConnector
from database.file_formats.performance.pageprogress import Locks
from omr.dataset import DatasetParams
from omr.experimenter.experimenter import SingleDataArgs, GlobalDataArgs, EvaluatorParams
from omr.steps.stafflines.detection.evaluator import EvaluationData
from omr.steps.stafflines.detection.experimenter import StaffLinesExperimenter
from database import DatabaseBook
from omr.dataset.datafiles import dataset_by_locked_pages, LockState
import logging

from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, SymbolMelodyEvaluator, precision_recall_f1, \
    Counts
from omr.steps.text.experimenter import TextExperimenter

# train, val = dataset_by_locked_pages(0.999, [LockState(Locks.SYMBOLS, True)], datasets=[b, c, d, e, f, g, h])
logger = logging.getLogger(__name__)


class Outpunt(NamedTuple):
    header_info: List[str]
    results: str
    meta: str


def evaluate_stafflines(pred_book: DatabaseBook, gt_book: DatabaseBook):
    def prepare_stafflines_gt(pred_book: DatabaseBook, gt_book: DatabaseBook):
        evaluation_data: List[EvaluationData] = []

        pcgts = []
        index = []
        for page in gt_book.pages_with_lock([LockState(Locks.STAFF_LINES, True)]):
            pcgts.append(PcGts.from_file(page.file('pcgts')))
        pages = [page.pcgts() for page in gt_book.pages(True)]
        l_of_pages = [page.page.location.page for page in pages]
        l_of_pcgts = [page.page.location.page for page in pcgts]
        index_of_pages = [l_of_pages.index(pcgt) for pcgt in l_of_pcgts]

        pcgts_pred = [page.pcgts() for page in pred_book.pages(True)]
        selected_pcgts = [pcgts_pred[i] for i in index_of_pages]

        for gt, pred in zip(pcgts, selected_pcgts):
            print(gt.page.location.page)
            print(pred.page.location.page)
            assert gt.page.location.page == pred.page.location.page
            evaluation_data.append(
                EvaluationData(gt.page.location.page, gt.page.all_music_lines(), pred.page.all_music_lines(),
                               (gt.page.image_height, gt.page.image_width)))

        return evaluation_data

    eval_data = prepare_stafflines_gt(pred_book, gt_book)
    s_data = SingleDataArgs(0, None, None, None, None, None)
    exp = StaffLinesExperimenter(s_data, logger)
    results = exp.evaluate(eval_data, EvaluatorParams(debug=False))
    # results = counts, prf1, (all_tp_staves, all_fp_staves, all_fn_staves)
    # print(results)
    output_string = exp.print_results(
        GlobalDataArgs("EXPERIMENT_OUT=", None, None, None, None, None, None, None, None, None, None, None, None, None,
                       None, None, None, None), [results], None)
    output_array = output_string[len("EXPERIMENT_OUT="):].split(",")
    excel_lines = []
    excel_lines.append(["Type", "Precision", "+-", "Recall", "+-", "F1", "+-"])
    excel_lines.append(["Staff Lines Detected"] + output_array[:6])
    excel_lines.append(["Staff lines hit"] + output_array[6:12])
    excel_lines.append(["Staves found"] + output_array[12:18])
    excel_lines.append(["Staff lines hit"] + output_array[18:24])
    return excel_lines


def evaluate_symbols(pred_book: DatabaseBook, gt_book: DatabaseBook):
    def prepare_symbol_gt(pred_book: DatabaseBook, gt_book: DatabaseBook):

        pcgts = []
        for page in gt_book.pages_with_lock([LockState(Locks.SYMBOLS, True)]):
            pcgts.append(PcGts.from_file(page.file('pcgts')))
        pages = [page.pcgts() for page in gt_book.pages(True)]
        l_of_pages = [page.page.location.page for page in pages]
        l_of_pcgts = [page.page.location.page for page in pcgts]
        index_of_pages = [l_of_pages.index(pcgt) for pcgt in l_of_pcgts]

        pcgts_pred = [page.pcgts() for page in pred_book.pages(True)]
        selected_pcgts = [pcgts_pred[i] for i in index_of_pages]
        pred_symbols = []
        gt_symbols = []
        for gt, pred in zip(pcgts, selected_pcgts):
            for p_line, gt_line in zip(pred.page.all_music_lines(), gt.page.all_music_lines()):
                pred_symbols.append(p_line.symbols)
                gt_symbols.append(gt_line.symbols)

        return pred_symbols, gt_symbols

    evaluator = SymbolDetectionEvaluator()
    pred_symbols, gt_symbols = prepare_symbol_gt(pred_book, gt_book)
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


def evaluate_text(pred_book: DatabaseBook, gt_book: DatabaseBook):
    def prepare_text_gt(pred_book: DatabaseBook, gt_book: DatabaseBook):

        pcgts = []
        for page in gt_book.pages_with_lock([LockState(Locks.TEXT, True)]):
            pcgts.append(PcGts.from_file(page.file('pcgts')))
        pages = [page.pcgts() for page in gt_book.pages(True)]
        l_of_pages = [page.page.location.page for page in pages]
        l_of_pcgts = [page.page.location.page for page in pcgts]
        index_of_pages = [l_of_pages.index(pcgt) for pcgt in l_of_pcgts]

        pcgts_pred = [page.pcgts() for page in pred_book.pages(True)]
        selected_pcgts = [pcgts_pred[i] for i in index_of_pages]
        pred_text = []
        gt_text = []
        for gt, pred in zip(pcgts, selected_pcgts):
            for p_line, gt_line in zip(pred.page.all_text_lines(), gt.page.all_text_lines()):
                pred_text.append(p_line.text())
                gt_text.append(gt_line.text())
                print(gt.page.location.page)
                print(pred.page.location.page)

                print(p_line.text())
                print(gt_line.text())
                print("__")

        return pred_text, gt_text

    pred_text, gt_text = prepare_text_gt(pred_book, gt_book)
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
    for i in excel_lines:
        print(i)
        print(len(i))
    return excel_lines


def evaluate_syllabels(pred_book: DatabaseBook, gt_book: DatabaseBook):
    def prepare_syllable_gt(pred_book: DatabaseBook, gt_book: DatabaseBook):

        pcgts = []
        for page in gt_book.pages_with_lock([LockState(Locks.TEXT, True)]):
            pcgts.append(PcGts.from_file(page.file('pcgts')))
        pages = [page.pcgts() for page in gt_book.pages(True)]
        l_of_pages = [page.page.location.page for page in pages]
        l_of_pcgts = [page.page.location.page for page in pcgts]
        index_of_pages = [l_of_pages.index(pcgt) for pcgt in l_of_pcgts]

        pcgts_pred = [page.pcgts() for page in pred_book.pages(True)]
        selected_pcgts = [pcgts_pred[i] for i in index_of_pages]

        def get_all_connections_of_music_line(line: Line, connections: List[Connection]):
            syl_connectors: List[SyllableConnector] = []
            for i in connections:
                if i.music_region.lines[0].id == line.id:  # Todo
                    syl_connectors += i.syllable_connections
                    pass
            return syl_connectors

        tp = 0
        fp = 0
        fn = 0
        for gt, pred in zip(pcgts, selected_pcgts):
            pred_annotations = pred.page.annotations
            gt_annotations = gt.page.annotations
            for p_line, gt_line in zip(pred.page.all_music_lines(), gt.page.all_music_lines()):
                syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations.connections)
                syl_connectors_gt = get_all_connections_of_music_line(gt_line, gt_annotations.connections)
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
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        acc = tp / (tp + fp + fn)
        return tp, fn ,fp , f1 , acc
    result = prepare_syllable_gt(pred_book, gt_book)
    excel_lines = []

    excel_lines.append(["Tp", "Fn", "FP", "F1", "acc"])
    excel_lines.append(result)
    return excel_lines

    # for p_line, gt_line in zip(pred.page.all_text_lines(), gt.page.all_text_lines()):
    #    pred_text.append(p_line.text())
    #    gt_text.append(gt_line.text())
    #    print(p_line.text())
    #    print(gt_line.text())
    #    print("__")

    # return pred_text, gt_text


if __name__ == "__main__":
    b = DatabaseBook('mulhouse_splitted')
    c = DatabaseBook('mulhouse_splitted_gt')
    excel_lines1 = evaluate_stafflines(b, c)
    excel_lines2 = evaluate_symbols(b, c)
    excel_lines3 = evaluate_text(b, c)
    excel_lines4 = evaluate_syllabels(b, c)

    from xlwt import Workbook

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    ind = 0
    for x in [excel_lines1, excel_lines2, excel_lines3, excel_lines4]:
        ind += 3
        for line in x:
            for ind1, cell in enumerate(line):
                sheet1.write(ind, ind1, str(cell))
            ind += 1
    wb.save("/tmp/eval_data.xls")

    pass
