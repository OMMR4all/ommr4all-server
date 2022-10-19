from dataclasses_json import dataclass_json, LetterCase

from database.file_formats.pcgts import *
from typing import List, Tuple, NamedTuple
import os
import numpy as np
from enum import IntEnum
from edit_distance import edit_distance
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
import prettytable
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from omr.experimenter.experimenter import EvaluatorParams
from omr.steps.symboldetection.predictor import SingleLinePredictionResult
from tools.dataset_statistics import drop_captial_of_text_line_center, para_text_of_text_line_center, \
    symbols_between_x1_x2, symbols_in_line, symbols_in_block


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Counts_Types:
    n_pages: int = 0
    n_staves: int = 0
    n_staff_lines: int = 0
    n_symbols: int = 0
    n_note_components: int = 0
    n_clefs: int = 0
    n_accids: int = 0
    n_symbols_in_text_area: int = 0
    n_symbols_in_text_area_fp: int = 0

    n_clef_after_drop_capital_area_all: int = 0
    n_clef_after_drop_capital_area_all_fp: int = 0

    n_clef_after_drop_capital_area_big: int = 0
    n_clef_after_drop_capital_area_big_fp: int = 0

    n_symbol_in_drop_capital_area: int = 0
    n_symbol_in_drop_capital_area_fp: int = 0

    n_symbol_above_para_text_area: int = 0
    n_symbol_above_para_text_area_fp: int = 0

    n_drop_capitals_all: int = 0
    n_drop_capitals_big: int = 0
    n_para_text: int = 0
    n_para_text2: int = 0
    n_para_drop_captial2: int = 0

    def to_np_array(self):
        return np.array(list(asdict(self).values()))


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Clef_Error_Types:
    n_clefs_tp: int = 0

    n_clefs_tp_pos: int = 0
    n_clefs_tp_type: int = 0

    n_clef_beginning_tp: int = 0
    n_clef_unspecific_tp: int = 0
    n_clef_after_drop_capital_area_all_tp: int = 0
    n_clef_after_drop_capital_area_big_tp: int = 0

    n_clefs_fp: int = 0
    n_clef_beginning_fp: int = 0
    n_clef_unspecific_fp: int = 0
    n_clef_after_drop_capital_area_all_fp: int = 0
    n_clef_after_drop_capital_area_big_fp: int = 0

    n_clefs_fn: int = 0
    n_clef_beginning_fn: int = 0
    n_clef_unspecific_fn: int = 0
    n_clef_after_drop_capital_area_all_fn: int = 0
    n_clef_after_drop_capital_area_big_fn: int = 0

    n_clef_type_fn_fp: int = 0
    n_clef_pos_fn_fp: int = 0


    def to_np_array(self):
        return np.array(list(asdict(self).values()))


class PRF2Metrics(IntEnum):
    SYMBOL = 0
    NOTE = 1
    NOTE_ALL = 2
    NOTE_PIS = 3
    CLEF = 4
    CLEF_ALL = 5
    CLEF_PIS = 6
    ACCID = 7

    COUNT = 8


class SymbolDetectionEvaluatorParams(NamedTuple):
    symbol_detected_min_distance: int = 5


class PRF2Index(IntEnum):
    P = 0
    R = 1
    F2 = 2

    COUNT = 3

    def __int__(self):
        return self.value


class Counts(IntEnum):
    TP = 0
    FP = 1
    FN = 2

    COUNT = 3

    def __int__(self):
        return self.value


class AccCounts(IntEnum):
    TRUE = 0
    FALSE = 1
    TOTAL = 2

    COUNT = 3

    def __int__(self):
        return self.value


class ConnectionCounter(NamedTuple):
    LOOPED = 0
    GAPED = 0
    NEUMESTART = 0


class ConfusionMatrix:

    def __init__(self):
        self.pairs = []
        self.fn = []
        self.fp = []

        self.cf_matrix = None

    def gather(self, pairs, gt_symbols, p_symbols):
        self.pairs += pairs
        self.fp += p_symbols
        self.fn += gt_symbols

    def plot_confusion_matrix(self, normalize=False, plot=False, conditional=False, show_tp=True):

        ## Generate Confusion Matrix
        y_true = []
        y_pred = []

        ## TP
        for (p_c, p_s), best in self.pairs:
            gt_c, gt_s = best
            gt_s: MusicSymbol = gt_s
            y_true.append(str(gt_s.graphical_connection).split('GraphicalConnectionType.')[1]
                          if gt_s.symbol_type == gt_s.symbol_type.NOTE else str(gt_s.symbol_type).split('SymbolType.')[
                1])
            y_pred.append(str(p_s.graphical_connection).split('GraphicalConnectionType.')[1]
                          if p_s.symbol_type == p_s.symbol_type.NOTE else str(p_s.symbol_type).split('SymbolType.')[1])

        ## FP
        for (p_c, p_s) in self.fp:
            p_s: MusicSymbol = p_s
            y_true.append('OTHER')
            y_pred.append(str(p_s.graphical_connection).split('GraphicalConnectionType.')[1]
                          if p_s.symbol_type == p_s.symbol_type.NOTE else str(p_s.symbol_type).split('SymbolType.')[1])

        ## FN
        for (p_c, p_s) in self.fn:
            p_s: MusicSymbol = p_s
            y_true.append(str(p_s.graphical_connection).split('GraphicalConnectionType.')[1]
                          if p_s.symbol_type == p_s.symbol_type.NOTE else str(p_s.symbol_type).split('SymbolType.')[1])
            y_pred.append('OTHER')

        labels = sorted(list(set(y_true + y_pred)))
        self.cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        if not show_tp:
            np.fill_diagonal(self.cf_matrix, 0)

        if normalize:
            self.cf_matrix = self.cf_matrix.astype('float') / self.cf_matrix.sum()
        if conditional:
            self.cf_matrix = self.cf_matrix.astype('float') / self.cf_matrix.sum(axis=1)[:, np.newaxis]

        cf_matrix_display = prettytable.PrettyTable()
        cf_matrix_display.add_column('Pred \ GT', labels + ['SUM'])

        for x in range(len(labels)):
            row = self.cf_matrix[x, :]
            cf_matrix_display.add_column(str(labels[x]), list(row) + [np.sum(row)])

        c_sum = []
        for x in range(len(labels)):
            column = self.cf_matrix[:, x]
            c_sum.append(np.sum(column))
        cf_matrix_display.add_column('SUM', c_sum + [np.sum(self.cf_matrix)])

        print(cf_matrix_display)

        def plot_confusion_matrix_pyplot(classes, cmap='Blues'):
            fig, ax = plt.subplots()
            im = ax.imshow(self.cf_matrix, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(self.cf_matrix.shape[1]),
                   yticks=np.arange(self.cf_matrix.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title='Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = self.cf_matrix.max() / 2.
            for i in range(self.cf_matrix.shape[0]):
                for j in range(self.cf_matrix.shape[1]):
                    ax.text(j, i, format(self.cf_matrix[i, j], fmt),
                            ha="center", va="center",
                            color="white" if self.cf_matrix[i, j] > thresh else "black")
            fig.tight_layout()
            return ax

        if plot:
            np.set_printoptions(precision=2)
            plot_confusion_matrix_pyplot(classes=labels)
            plt.show()


def precision_recall_f1(tp, fp, fn) -> Tuple[float, float, float]:
    if tp == 0:
        return 0.0, 0.0, 0.0
    return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


class SequenceDiffs(NamedTuple):
    missing_notes: int = 0
    wrong_note_connections: int = 0
    wrong_position_in_staff: int = 0
    missing_clefs: int = 0
    missing_accids: int = 0

    additional_note: int = 0
    add_wrong_note_con: int = 0
    add_wrong_pos_in_staff: int = 0
    additional_clef: int = 0
    additional_accid: int = 0

    total: int = 0
    total_errors: int = 0


class Codec:
    def __init__(self):
        self.codec = []
        self.neume_codec = []

    def get(self, v):
        if v in self.codec:
            return self.codec.index(v)
        self.codec.append(v)
        return len(self.codec) - 1

    def get_neume(self, v):
        if v in self.codec:
            return self.codec.index(v)
        self.codec.append(v)
        return len(self.codec) - 1

    def label_to_neume_sequence(self, sequence: List):
        neumes = []
        last_no_neume = True
        for symbol in sequence:
            symbol = self.codec[symbol]
            symbol_type = symbol[0]
            if symbol_type != SymbolType.NOTE:
                last_no_neume = True
                continue

            if symbol[3] == GraphicalConnectionType.NEUME_START or last_no_neume:
                last_no_neume = False
                neumes.append([symbol])
            else:
                neumes[-1].append(symbol)

        neumes = map(tuple, neumes)

        return list(map(self.get_neume, neumes))

    def symbols_to_label_sequence(self, symbols: List[MusicSymbol], note_connection_type: bool):
        sequence = []
        for symbol in symbols:
            if symbol.symbol_type == SymbolType.ACCID:
                sequence.append((symbol.symbol_type, symbol.accid_type))
            elif symbol.symbol_type == SymbolType.CLEF:
                sequence.append((symbol.symbol_type, symbol.clef_type, symbol.position_in_staff))
            elif symbol.symbol_type == SymbolType.NOTE:
                if note_connection_type:
                    sequence.append(
                        (symbol.symbol_type, symbol.note_type, symbol.position_in_staff, symbol.graphical_connection))
                else:
                    sequence.append((symbol.symbol_type, symbol.note_type, symbol.position_in_staff, True))
            else:
                raise Exception('Unknown symbol type')

        return list(map(self.get, sequence))

    def symbols_to_melody_sequence(self, symbols: List[MusicSymbol]):
        sequence = []
        for symbol in symbols:
            if symbol.symbol_type == SymbolType.NOTE:
                sequence.append((symbol.note_name, symbol.octave))
        return sequence

    def compute_sequence_diffs(self, gt, pred) -> SequenceDiffs:
        sm = SequenceMatcher(a=pred, b=gt, autojunk=False, isjunk=False)
        total = max(len(gt), len(pred))
        missing_accids = 0
        missing_notes = 0
        missing_clefs = 0
        wrong_note_connections = 0
        wrong_position_in_staff = 0

        additional_note = 0
        add_wrong_pos_in_staff = 0
        add_wrong_note_con = 0
        additional_clef = 0
        additional_accid = 0

        total_errors = 0
        true_positives = 0
        # print(list(map(self.codec.__getitem__, pred)))
        # print(list(map(self.codec.__getitem__, gt)))
        # print(sm.get_opcodes())
        for opcode, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes():
            if opcode == 'equal':
                true_positives += gt_end - gt_start
            elif opcode == 'insert' or opcode == 'replace' or opcode == 'delete':
                total_errors += pred_end - pred_start + gt_end - gt_start
                for i, s in enumerate(gt[gt_start:gt_end]):
                    entry = self.codec[s]
                    symbol_type = entry[0]
                    if symbol_type == SymbolType.ACCID:
                        missing_accids += 1
                    elif symbol_type == SymbolType.NOTE:
                        if opcode == 'replace' and pred_end > pred_start + i:
                            # check for wrong connection
                            p = self.codec[pred[pred_start + i]]
                            if p[0] == symbol_type:
                                if p[3] == entry[3]:
                                    wrong_position_in_staff += 1
                                else:
                                    wrong_note_connections += 1
                            else:
                                missing_notes += 1
                        else:
                            missing_notes += 1
                    elif symbol_type == SymbolType.CLEF:
                        missing_clefs += 1
                    else:
                        raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

                for i, s in enumerate(pred[pred_start:pred_end]):
                    entry = self.codec[s]
                    symbol_type = entry[0]
                    if symbol_type == SymbolType.ACCID:
                        additional_accid += 1
                    elif symbol_type == SymbolType.NOTE:
                        if opcode == 'replace' and gt_end > gt_start + i:
                            # check for wrong connection
                            p = self.codec[gt[gt_start + i]]
                            if p[0] == symbol_type:
                                if p[3] == entry[3]:
                                    add_wrong_pos_in_staff += 1
                                else:
                                    add_wrong_note_con += 1
                            else:
                                additional_note += 1
                        else:
                            additional_note += 1
                    elif symbol_type == SymbolType.CLEF:
                        additional_clef += 1
                    else:
                        raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

            else:
                raise ValueError(opcode)

        return SequenceDiffs(missing_notes, wrong_note_connections, wrong_position_in_staff, missing_clefs,
                             missing_accids,
                             additional_note, add_wrong_note_con, add_wrong_pos_in_staff,
                             additional_clef, additional_accid,
                             total, total_errors)


class SymbolDetectionEvaluator:
    def __init__(self, params: EvaluatorParams = None):
        self.params = params if params else EvaluatorParams()
        self.codec = Codec()

    def evaluate(self, gt_symbols: List[List[MusicSymbol]], pred_symbols: List[List[MusicSymbol]]):
        cm = ConfusionMatrix()
        min_distance_sqr = self.params.symbol_detected_min_distance ** 2

        def extract_coords_of_symbols(symbols: List[MusicSymbol]) -> List[Tuple[Point, MusicSymbol]]:
            return [(s.coord, s) for s in symbols]

        f_metrics = np.zeros((0, PRF2Metrics.COUNT, PRF2Metrics.COUNT), dtype=float)
        acc_metrics = np.zeros((0, 4), dtype=float)
        counts = np.zeros((0, 5, Counts.COUNT), dtype=int)

        acc_counts = np.zeros((0, 8, AccCounts.COUNT), dtype=int)

        total_diffs = np.zeros(12, dtype=int)

        for gt, pred in zip(gt_symbols, pred_symbols):
            gt_sequence = self.codec.symbols_to_label_sequence(gt, False)
            pred_sequence = self.codec.symbols_to_label_sequence(pred, False)
            gt_sequence_nc = self.codec.symbols_to_label_sequence(gt, True)
            pred_sequence_nc = self.codec.symbols_to_label_sequence(pred, True)
            neume_gt_sequence = self.codec.label_to_neume_sequence(gt_sequence_nc)
            neume_pred_sequence = self.codec.label_to_neume_sequence(pred_sequence_nc)
            sequence_ed = edit_distance(gt_sequence, pred_sequence)
            # print(gt_sequence)
            # print(pred_sequence)
            # print(sequence_ed)
            sequence_ed_nc = edit_distance(gt_sequence_nc, pred_sequence_nc)
            neume_sequence_ed = edit_distance(neume_gt_sequence, neume_pred_sequence)
            diffs = np.asarray(self.codec.compute_sequence_diffs(gt_sequence_nc, pred_sequence_nc))
            total_diffs += diffs
            p_symbols = extract_coords_of_symbols(pred)
            p_symbols_orig = p_symbols[:]
            gt_symbols_orig = extract_coords_of_symbols(gt)
            gt_symbols = gt_symbols_orig[:]
            pairs = []

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
            n_tp = len(pairs)
            n_fp = len(p_symbols)
            n_fn = len(gt_symbols)

            cm.gather(pairs, gt_symbols, p_symbols)
            if n_tp == 0 and n_fp == 0 and n_fn == 0:
                # empty
                print("Empty. Skipping!")
                continue

            def sub_group(symbol_types: List[SymbolType]):
                l_tp = [(p, gt) for (_, p), (_, gt) in pairs if
                        gt.symbol_type in symbol_types and p.symbol_type in symbol_types]

                l_fp = [p for (_, p), (_, gt) in pairs if
                        gt.symbol_type not in symbol_types and p.symbol_type in symbol_types] \
                       + [p for (_, p) in p_symbols if p.symbol_type in symbol_types]

                l_fn = \
                    [gt for (_, p), (_, gt) in pairs if
                     p.symbol_type not in symbol_types and gt.symbol_type in symbol_types] \
                    + [gt for (_, gt) in gt_symbols if gt.symbol_type in symbol_types]

                tp, fp, fn = tuple(list(map(len, (l_tp, l_fp, l_fn))))

                try:
                    return (tp, fp, fn), precision_recall_f1(tp, fp, fn), (l_tp, l_fp, l_fn)
                except ZeroDivisionError:
                    return (tp, fp, fn), None, (l_tp, l_fp, l_fn)

            def note_sub_group(lists, prf2metric: PRF2Metrics):
                l_tp, _, _ = lists
                if prf2metric == PRF2Metrics.NOTE_ALL:
                    l_true = [(p, gt) for p, gt in l_tp if p.graphical_connection == gt.graphical_connection]
                    l_false = [(p, gt) for p, gt in l_tp if gt.graphical_connection != p.graphical_connection]
                elif prf2metric == PRF2Metrics.NOTE_PIS:
                    l_true = [(p, gt) for p, gt in l_tp if p.position_in_staff == gt.position_in_staff]
                    l_false = [(p, gt) for p, gt in l_tp if p.position_in_staff != gt.position_in_staff]

                true, false = tuple(list(map(len, (l_true, l_false))))
                try:
                    return (true, false, true + false), true / (true + false), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, true + false), None, (l_true, l_false, [])

            def clef_sub_group(lists, prf2metric: PRF2Metrics):
                l_tp, _, _ = lists
                if prf2metric == PRF2Metrics.CLEF_ALL:
                    l_true = [(p, gt) for p, gt in l_tp if gt.clef_type == p.clef_type]
                    l_false = [(p, gt) for p, gt in l_tp if gt.clef_type != p.clef_type]
                elif prf2metric == PRF2Metrics.CLEF_PIS:
                    l_true = [(p, gt) for p, gt in l_tp if p.position_in_staff == gt.position_in_staff]
                    l_false = [(p, gt) for p, gt in l_tp if p.position_in_staff != gt.position_in_staff]

                true, false = tuple(list(map(len, (l_true, l_false))))
                try:
                    return (true, false, true + false), true / (false + true), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, true + false), None, (l_true, l_false, [])

            def accid_sub_group(lists, prf2metric: PRF2Metrics):
                l_tp, _, _ = lists
                l_true = [(p, gt) for p, gt in l_tp if gt.accid_type == p.accid_type]
                l_false = [(p, gt) for p, gt in l_tp if gt.accid_type != p.accid_type]
                true, false = tuple(list(map(len, (l_true, l_false))))
                try:
                    return (true, false, true + false), true / (false + true), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, true + false), None, (l_true, l_false, [])


            all_counts, all_metrics, all_ = sub_group([SymbolType.NOTE, SymbolType.ACCID, SymbolType.CLEF])
            note_counts, note_metrics, notes = sub_group([SymbolType.NOTE])
            clef_counts, clef_metrics, clefs = sub_group([SymbolType.CLEF])

            accid_counts, accid_metrics, accids = sub_group([SymbolType.ACCID])

            note_all_counts, note_all_metrics, note_all = note_sub_group(notes, PRF2Metrics.NOTE_ALL)
            note_pis_counts, note_pis_metrics, note_pis = note_sub_group(notes, PRF2Metrics.NOTE_PIS)

            clef_all_counts, clef_all_metrics, clefs_all = clef_sub_group(clefs, PRF2Metrics.CLEF_ALL)
            clef_pis_counts, clef_pis_metrics, clefs_pis = clef_sub_group(clefs, PRF2Metrics.CLEF_PIS)

            accid_all_counts, accid_all_metrics, accid_all = accid_sub_group(accids, PRF2Metrics.ACCID)

            counts = np.concatenate((counts, [[[n_tp, n_fp, n_fn],
                                               all_counts,
                                               note_counts,
                                               clef_counts,
                                               accid_counts,
                                               ]]), axis=0)

            acc_counts = np.concatenate((acc_counts, [[
                note_all_counts,
                note_pis_counts,
                clef_all_counts,
                clef_pis_counts,
                accid_all_counts,
                (sequence_ed[1], sequence_ed[0], sum(sequence_ed)),
                (sequence_ed_nc[1], sequence_ed_nc[0], sum(sequence_ed_nc)),
                (neume_sequence_ed[1], neume_sequence_ed[0], sum(neume_sequence_ed))
            ]]), axis=0)

            acc_metrics = np.concatenate((acc_metrics, [[
                note_all_metrics,
                note_pis_metrics,
                clef_all_metrics,
                clef_pis_metrics,
            ]]), axis=0)

        acc_counts = acc_counts.sum(axis=0)
        acc_acc = (acc_counts[:, AccCounts.TRUE] / acc_counts[:, AccCounts.TOTAL]).reshape((-1, 1))

        # normalize errors
        total_diffs_count = total_diffs
        total_diffs = total_diffs / total_diffs[-1]
        # transfer total / errors => acc = 1 - errors / total
        total_diffs[-2] = 1 - 1 / total_diffs[-2]
        cm.plot_confusion_matrix()
        cm.plot_confusion_matrix(normalize=True)
        return f_metrics.mean(axis=0), counts.sum(axis=0), acc_counts, acc_acc, total_diffs, total_diffs_count


class SymbolMelodyEvaluator:
    def __init__(self, params: EvaluatorParams = None):
        self.params = params if params else EvaluatorParams()
        self.codec = Codec()

    def evaluate(self, gt_symbols: List[List[MusicSymbol]], pred_symbols: List[List[MusicSymbol]]):
        acc_counts1 = np.zeros((0, 1, AccCounts.COUNT), dtype=int)

        for gt, prediction in zip(gt_symbols, pred_symbols):
            gt_sequence = self.codec.symbols_to_melody_sequence(gt)
            pred_sequence = self.codec.symbols_to_melody_sequence(prediction)
            sequence_ed = edit_distance(gt_sequence, pred_sequence)
            to_ocncatendate = np.array([[
                (sequence_ed[1], sequence_ed[0], sum(sequence_ed)),
            ]])
            acc_counts1 = np.concatenate((acc_counts1, [[
                (sequence_ed[1], sequence_ed[0], sum(sequence_ed)),
            ]]), axis=0)
        acc_counts1 = acc_counts1.sum(axis=0)
        acc_acc1 = (acc_counts1[:, AccCounts.TRUE] / acc_counts1[:, AccCounts.TOTAL]).reshape((-1, 1))

        return acc_counts1, acc_acc1


class SymbolErrorTypeDetectionEvaluator:
    def __init__(self, params: EvaluatorParams = None):
        self.params = params if params else EvaluatorParams()

    def evaluate(self, predictions):
        counts = Counts_Types()
        counts2 = Clef_Error_Types()
        min_distance_sqr = self.params.symbol_detected_min_distance ** 2

        def extract_coords_of_symbols(symbols: List[MusicSymbol]) -> List[Tuple[Point, MusicSymbol]]:
            return [(s.coord, s) for s in symbols]

        for ind, page_pred in enumerate(predictions):

            page = page_pred.pcgts.page
            # print(page.location.page)
            paragraph = page.blocks_of_type(BlockType.PARAGRAPH)
            drop_capitals = page.blocks_of_type(BlockType.DROP_CAPITAL)
            counts.n_para_drop_captial2 += len(drop_capitals)
            counts.n_para_text2 += len(page.blocks_of_type(BlockType.PARAGRAPH))

            for ml in page_pred.music_lines:
                pairs = []
                pair_dict = {}
                symbols = ml.symbols
                line = ml.line.operation.music_line
                # print(ml.line.operation.music_line.id)
                gt_symbols = ml.line.operation.music_line.symbols

                pred = symbols[:]
                gt = gt_symbols[:]
                p_symbols = extract_coords_of_symbols(pred)
                gt_symbols_orig = extract_coords_of_symbols(gt)
                gt_symbols = gt_symbols_orig[:]
                b_tl = page.closest_below_text_line_to_music_line(line, True)
                a_tl = page.closest_above_text_line_to_music_line(line, True)
                d_capitals = drop_captial_of_text_line_center(b_tl, line,
                                                              drop_capitals)  ## p.drop_capitals_of_text_line(b_tl)

                def clefs_type_count(lists, symbol_types):
                    l_tp = [(p, gt) for (_, p), (_, gt) in pairs if
                            gt.symbol_type in symbol_types and p.symbol_type in symbol_types]

                    l_fp = [(p, True) for (_, p), (_, gt) in pairs if
                            gt.symbol_type not in symbol_types and p.symbol_type in symbol_types] \
                           + [(p, False) for (_, p) in p_symbols if p.symbol_type in symbol_types]

                    l_fn = \
                        [(gt, True) for (_, p), (_, gt) in pairs if
                         p.symbol_type not in symbol_types and gt.symbol_type in symbol_types] \
                        + [(gt, False) for (_, gt) in gt_symbols if gt.symbol_type in symbol_types]

                    l_fp_type = [pad for pad in l_fp if pad[1] == True]
                    l_fp_not_type = [pad for pad in l_fp if pad[1] == False]
                    l_fn_type = [pad for pad in l_fn if pad[1] == True]
                    l_fn_not_type = [pad for pad in l_fn if pad[1] == False]
                    clef_l_true = [(p, gt) for p, gt in l_tp if gt.clef_type == p.clef_type]
                    clef_l_false = [(p, gt) for p, gt in l_tp if gt.clef_type != p.clef_type]
                    pis_l_true = [(p, gt) for p, gt in l_tp if p.position_in_staff == gt.position_in_staff]
                    pis_l_false = [(p, gt) for p, gt in l_tp if p.position_in_staff != gt.position_in_staff]
                    count_pis = len(pis_l_false)
                    count_type = len(clef_l_false)
                    # count_type_fp = len(l_fp_type)
                    # count_type_fn = len(l_fn_type)
                    count_beginning_fn = [s[0] for s in l_fn if s[0].id == gt_symbols_orig[0][1].id]
                    count_beginning_fp = [s[0] for s in l_fn if s[0].id == pred_symbols[0][1].id]

                    tp, fp, fn = tuple(list(map(len, (l_tp, l_fp, l_fn))))
                    l_tp, l_fp, l_fn = lists
                    count_tp = 0
                    count_type_fp = len(l_fp)
                    count_type_fn = len(l_fn)

                    tp = len(pis_l_true) + len(clef_l_true)
                    clef_l_start = [(p, gt) for p in l_tp if gt.clef_type == p.clef_type]

                    return (true, false, true + false), None, (l_true, l_false, [])

                def symbol_after_x(x, symbols: List[MusicSymbol]):
                    for symbol_1 in symbols:
                        if symbol_1.coord.x > x:
                            return symbol_1
                    return None

                para_text = para_text_of_text_line_center(b_tl, line, paragraph)
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
                        pair_dict[p_s.id] = best_s[1]
                        # pair_dict[best_s[1].s_id] = p_s

                        del gt_symbols[best_gti]
                        del p_symbols[p_i]
                n_tp = len(pairs)
                n_fp = len(p_symbols)
                n_fn = len(gt_symbols)

                def clefs_stats(counts: Clef_Error_Types):
                    pass

                def check_if_clef_after_dp(symbols: List[MusicSymbol], d_capitals, symbol: MusicSymbol):
                    def is_big_dc(dp, min_size=0.09):
                        if cap.aabb.bottom() - cap.aabb.top() > min_size:
                            return True
                        else:
                            return False

                    for cap in d_capitals:
                        center_x = cap.aabb.center.x

                        next_symbol = symbol_after_x(center_x, symbols)
                        if next_symbol and next_symbol.id == symbol.id:
                            return True, is_big_dc(cap)
                    return False, False

                def is_system_beginning(symbol: MusicSymbol, symbols: List[MusicSymbol]):
                    if symbol.id == symbols[0].id:
                        return True
                    else:
                        return False

                for coord, symbol in gt_symbols:
                    if symbol.symbol_type == symbol.symbol_type.CLEF:
                        counts2.n_clefs_fn += 1
                        after_dp, after_dp_size = check_if_clef_after_dp(gt, d_capitals, symbol)
                        if after_dp:
                            counts2.n_clef_after_drop_capital_area_all_fn += 1
                            if after_dp_size:
                                counts2.n_clef_after_drop_capital_area_big_fn += 1
                        elif is_system_beginning(symbol, gt):
                            counts2.n_clef_beginning_fn += 1
                        else:
                            counts2.n_clef_unspecific_fn += 1

                for coord, symbol in p_symbols:

                    if symbol.symbol_type == symbol.symbol_type.CLEF:
                        counts2.n_clefs_fp += 1
                        after_dp, after_dp_size = check_if_clef_after_dp(pred, d_capitals, symbol)
                        if after_dp:

                            counts2.n_clef_after_drop_capital_area_all_fp += 1
                            if after_dp_size:
                                counts2.n_clef_after_drop_capital_area_big_fp += 1
                        elif is_system_beginning(symbol, pred):

                            counts2.n_clef_beginning_fp += 1
                        else:

                            counts2.n_clef_unspecific_fp += 1

                for predict, gtruth in pairs:
                    symbol = predict[1]
                    if symbol.symbol_type == symbol.symbol_type.CLEF:
                        if symbol.clef_type != gtruth[1].clef_type or symbol.position_in_staff != gtruth[1].position_in_staff:
                            if symbol.clef_type != gtruth[1].clef_type:
                                counts2.n_clef_type_fn_fp += 1
                            else:
                                counts2.n_clef_pos_fn_fp += 1
                        else:
                            counts2.n_clefs_tp += 1
                            if gtruth[1].position_in_staff == symbol.position_in_staff:
                                counts2.n_clefs_tp_pos += 1
                            elif gtruth[1].symbol_type == symbol.symbol_type:
                                counts2.n_clefs_tp_type += 1

                            after_dp, after_dp_size = check_if_clef_after_dp(pred, d_capitals, symbol)
                            if after_dp:
                                counts2.n_clef_after_drop_capital_area_all_tp += 1
                                if after_dp_size:
                                    counts2.n_clef_after_drop_capital_area_big_tp += 1
                            elif is_system_beginning(symbol, pred):
                                counts2.n_clef_beginning_tp += 1
                            else:
                                counts2.n_clef_unspecific_tp += 1

                # page = ml.line.operation.page
                # counts.n_pages += 1
                # mls = p.all_music_lines()
                # counts.n_staves += len(mls)

                # print(pcgts.dataset_page().page)

                # para_text = p.para_text_of_text_line(b_tl)
                symbols_above_para_text = []
                for para in para_text:
                    ss = symbols_between_x1_x2(symbols, para.aabb.left(), para.aabb.right())
                    # if len(ss) > 0:
                    #    print(p.location.page)

                    symbols_above_para_text += ss
                # 1
                symbols_above_para_text_tp = [i for i in symbols_above_para_text if i.id in pair_dict]

                for cap in d_capitals:
                    center_x = cap.aabb.center.x

                    next_symbol = symbol_after_x(center_x, symbols)
                    if next_symbol is not None:
                        if next_symbol.symbol_type == SymbolType.CLEF:
                            is_pair = True if next_symbol.id in pair_dict else False
                            counts.n_clef_after_drop_capital_area_all += 1
                            if is_pair is False:
                                counts.n_clef_after_drop_capital_area_all_fp += 1
                            if cap.aabb.bottom() - cap.aabb.top() > 0.09:
                                counts.n_clef_after_drop_capital_area_big += 1
                                if is_pair is False:
                                    counts.n_clef_after_drop_capital_area_big_fp += 1
                                counts.n_drop_capitals_big += 1

                symbols_in_text_area = symbols_in_line(symbols, a_tl)
                symbols_in_text_area += symbols_in_line(symbols, b_tl)
                symbols_in_text_area_tp = [i for i in symbols_in_text_area if i.id in pair_dict]

                # if len(symbols_in_text_area) > 0:
                #    print(pcgts.dataset_page().page)
                symbols_in_drop_capital = []
                for d in d_capitals:
                    symbols_in_drop_capital += symbols_in_block(symbols, d)
                # 1
                symbols_in_drop_capital_tp = [i for i in symbols_in_drop_capital if i.id in pair_dict]
                counts.n_staff_lines += len(line.staff_lines)
                counts.n_symbols += len(symbols)
                counts.n_note_components += len([s for s in symbols if s.symbol_type == SymbolType.NOTE])
                counts.n_clefs += len([s for s in symbols if s.symbol_type == SymbolType.CLEF])
                counts.n_accids += len([s for s in symbols if s.symbol_type == SymbolType.ACCID])
                counts.n_drop_capitals_all += len(d_capitals)
                counts.n_para_text += len(para_text)
                counts.n_symbol_above_para_text_area += len(symbols_above_para_text)
                counts.n_symbols_in_text_area += len(symbols_in_text_area)
                counts.n_symbol_in_drop_capital_area += len(symbols_in_drop_capital)

                counts.n_symbols_in_text_area_fp += (len(symbols_in_text_area) - len(symbols_in_text_area_tp))
                counts.n_symbol_in_drop_capital_area_fp += (
                            len(symbols_in_drop_capital) - len(symbols_in_drop_capital_tp))

                counts.n_symbol_above_para_text_area_fp += len(symbols_above_para_text) - len(
                    symbols_above_para_text_tp)
        # pt = PrettyTable([n for n, _ in counts.to_dict().items()])
        # pt.add_row([v for _, v in counts.to_dict().items()])
        # print(pt)

        return counts, counts2


if __name__ == '__main__':
    from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
    from omr.steps.symboldetection.pixelclassifier.predictor import PCPredictor
    from omr.steps.symboldetection.pixelclassifier.meta import Meta

    from prettytable import PrettyTable
    from database import DatabaseBook
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    b = DatabaseBook('Pa_14819_gt')
    eval_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()]

    # eval_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[12:13]]
    # print([e.page.location.local_path() for e in eval_pcgts])
    pred = PCPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    # ps = list(pred.predict([p.page.location for p in val_pcgts]))
    # predictor = create_predictor(
    #    PredictorTypes.PIXEL_CLASSIFIER,
    #    SymbolDetectionPredictorParameters([b.local_path(os.path.join('pc_symbol_detection', 'model'))]))
    ps = [p.page.location for p in eval_pcgts]
    for p in ps:
        print(p.page)

    gt_symbols, pred_symbols = [], []
    predictions = []
    raw_gt = []
    raw_pred = []
    for p in pred.predict(ps):
        for music_line in p.music_lines:
            pred_symbols.append(music_line.symbols)
            gt_symbols.append(music_line.line.operation.music_line.symbols)
            raw_gt += music_line.line.operation.music_line.symbols
            raw_pred += music_line.symbols

        predictions.append(p)

    evaluator = SymbolDetectionEvaluator()
    metrics, counts, acc_counts, acc_acc, diffs = evaluator.evaluate(gt_symbols, pred_symbols)

    evaluator2 = SymbolErrorTypeDetectionEvaluator()
    counts1, counts2 = evaluator2.evaluate(predictions)
    pt = PrettyTable([n for n, _ in counts1.to_dict().items()])
    pt.add_row([v for _, v in counts1.to_dict().items()])
    print(len(raw_gt))
    print(len(raw_pred))

    at = PrettyTable()

    at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])
    at.add_column("TP", counts[:, Counts.TP])
    at.add_column("FP", counts[:, Counts.FP])
    at.add_column("FN", counts[:, Counts.FN])

    prec_rec_f1 = np.array([precision_recall_f1(c[Counts.TP], c[Counts.FP], c[Counts.FN]) for c in counts])
    at.add_column("Precision", prec_rec_f1[:, 0])
    at.add_column("Recall", prec_rec_f1[:, 1])
    at.add_column("F1", prec_rec_f1[:, 2])
    print(at.get_string())

    # print(acc_counts[:, AccCounts.FALSE])
    # print(AccCounts.TRUE.value)

    at = PrettyTable()
    at.add_column("Type", ["Note all", "Note PIS", "Clef All", "Clef PIS", "Accid All", "Sequence", "Sequence NC",
                           "Sequence Neume"])

    at.add_column("True", acc_counts[:, AccCounts.TRUE])
    at.add_column("False", acc_counts[:, AccCounts.FALSE])
    at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
    at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

    print(at.get_string())

    import xlwt
    from xlwt import Workbook

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    types = ["Type", "Note all", "Note PIS", "Clef All", "Clef PIS", "Accid All", "Sequence", "Sequence NC",
             "Sequence Neume"]
    x, y = 0, 0
    for xy in types:
        sheet1.write(y, x, xy)
        x += 1
    y += 1
    x = 0
    for xy in ["True"] + list(acc_counts[:, AccCounts.TRUE]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:
                sheet1.write(y, x, float(xy))
        x += 1
    x = 0
    y += 1

    for xy in ["False"] + list(acc_counts[:, AccCounts.FALSE]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:
                sheet1.write(y, x, float(xy))
        x += 1
    x = 0
    y += 1

    for xy in ["Total"] + list(acc_counts[:, AccCounts.TOTAL]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:
                sheet1.write(y, x, float(xy))
        x += 1
    x = 0
    y += 1
    for xy in ["Accuracy [%]"] + list(acc_acc[:, 0] * 100):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:
                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    y += 1
    types = ["Type", "All", "All", "Notes", "Clefs", "Accids"]
    x = 0

    for xy in types:
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["TP"] + list(counts[:, Counts.TP]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["FP"] + list(counts[:, Counts.FP]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["FN"] + list(counts[:, Counts.FN]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["Precision"] + list(prec_rec_f1[:, 0]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["Recall"] + list(prec_rec_f1[:, 1]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0
    for xy in ["F1"] + list(prec_rec_f1[:, 2]):
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    y += 1
    x = 0

    for xy in [n for n, _ in counts1.to_dict().items()]:
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    y += 1
    x = 0

    for xy in [v for _, v in counts1.to_dict().items()]:
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        x += 1
    pt = PrettyTable([n for n, _ in counts2.to_dict().items()])
    pt.add_row([v for _, v in counts2.to_dict().items()])
    y += 1
    y += 1

    y_before = y
    y += 1
    x = 0
    for xy in [n for n, _ in counts2.to_dict().items()]:
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        y += 1

    y = y_before
    x = 1
    for xy in [v for _, v in counts2.to_dict().items()]:
        if isinstance(xy, str):
            sheet1.write(y, x, xy)
        else:
            if np.isnan(xy):
                sheet1.write(y, x, "None")
            else:

                sheet1.write(y, x, float(xy))
        y += 1
    wb.save('wopp5.xls')
