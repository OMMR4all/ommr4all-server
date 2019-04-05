from database.file_formats.pcgts import *
from typing import List, Tuple, NamedTuple
import os
import numpy as np
from enum import IntEnum
from edit_distance import edit_distance

class PRF2Metrics(IntEnum):
    SYMBOL = 0
    NOTE = 1
    NOTE_ALL = 2
    NOTE_GC = 3
    NOTE_NS = 5
    NOTE_PIS = 6
    CLEF = 7
    CLEF_ALL = 8
    CLEF_PIS = 9
    ACCID = 10

    COUNT = 11


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


def precision_recall_f1(tp, fp, fn) -> Tuple[float, float, float]:
    if tp == 0:
        return 0.0, 0.0, 0.0
    return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


class Codec:
    def __init__(self):
        self.codec = []

    def get(self, v):
        if v in self.codec:
            return self.codec.index(v)
        self.codec.append(v)
        return len(self.codec) - 1

    def symbols_to_label_sequence(self, symbols: List[Symbol], note_connection_type: bool):
        sequence = []
        for symbol in symbols:
            if symbol.symbol_type == SymbolType.ACCID:
                a: Accidental = symbol
                sequence.append((a.symbol_type, a.accidental))
            elif symbol.symbol_type == SymbolType.CLEF:
                c: Clef = symbol
                sequence.append((c.symbol_type, c.clef_type, c.position_in_staff))
            elif symbol.symbol_type == SymbolType.NEUME:
                n: Neume = symbol
                for nc in n.notes:
                    if not note_connection_type:
                        sequence.append((n.symbol_type, nc.note_type, nc.position_in_staff))
                    else:
                        sequence.append((n.symbol_type, nc.note_type, nc.position_in_staff, nc.graphical_connection))
            else:
                raise Exception('Unknown symbol type')

        return list(map(self.get, sequence))


class SymbolDetectionEvaluatorParams(NamedTuple):
    symbol_detected_min_distance: int = 5


class SymbolDetectionEvaluator:
    def __init__(self, params: SymbolDetectionEvaluatorParams = None):
        self.params = params if params else SymbolDetectionEvaluatorParams()
        self.codec = Codec()

    def evaluate(self, gt_symbols: List[List[Symbol]], pred_symbols: List[List[Symbol]]):
        min_distance_sqr = self.params.symbol_detected_min_distance ** 2
        def extract_symbol_coord(s: Symbol) -> List[Tuple[Point, Symbol]]:
            if s.symbol_type == SymbolType.NEUME:
                n: Neume = s
                out = [(nc.coord, nc) for nc in n.notes]
                out[0][1].neume_start = True
                out[0][1].graphical_connection = GraphicalConnectionType.GAPED
                for _, nc in out[1:]:
                    nc.neume_start = False
                return out
            elif s.symbol_type == SymbolType.NOTE_COMPONENT:
                nc: NoteComponent = s
                return [(nc.coord, nc)]
            elif s.symbol_type == SymbolType.CLEF:
                c: Clef = s
                return [(c.coord, c)]
            elif s.symbol_type == SymbolType.ACCID:
                a: Accidental = s
                return [(a.coord, a)]
            else:
                raise Exception('Unknown symbol type {}'.format(s.symbol_type))

        def extract_coords_of_symbols(symbols: List[Symbol]) -> List[Tuple[Point, Symbol]]:
            l = []
            for s in symbols:
                l += extract_symbol_coord(s)

            return l

        f_metrics = np.zeros((0, PRF2Metrics.COUNT, PRF2Metrics.COUNT), dtype=float)
        acc_metrics = np.zeros((0, 6), dtype=float)
        counts = np.zeros((0, 5, Counts.COUNT), dtype=int)

        acc_counts = np.zeros((0, 9, AccCounts.COUNT), dtype=int)

        for gt, pred in zip(gt_symbols, pred_symbols):
            gt_sequence = self.codec.symbols_to_label_sequence(gt, False)
            pred_sequence = self.codec.symbols_to_label_sequence(pred, False)
            gt_sequence_nc = self.codec.symbols_to_label_sequence(gt, True)
            pred_sequence_nc = self.codec.symbols_to_label_sequence(pred, True)
            sequence_ed = edit_distance(gt_sequence, pred_sequence)
            sequence_ed_nc = edit_distance(gt_sequence_nc, pred_sequence_nc)
            pairs = []
            p_symbols = extract_coords_of_symbols(pred)
            gt_symbols_orig = extract_coords_of_symbols(gt)
            gt_symbols = gt_symbols_orig[:]

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

            if n_tp == 0 and n_fp == 0 and n_fn == 0:
                # empty
                print("Empty. Skipping!")
                continue

            def sub_group(symbol_types: List[SymbolType]):
                l_tp = [(p, gt) for (_, p), (_, gt) in pairs if gt.symbol_type in symbol_types and p.symbol_type in symbol_types]

                l_fp = [p for (_, p), (_, gt) in pairs if gt.symbol_type not in symbol_types and p.symbol_type in symbol_types] \
                        + [p for (_, p) in p_symbols if p.symbol_type in symbol_types]

                l_fn = \
                    [gt for (_, p), (_, gt) in pairs if p.symbol_type not in symbol_types and gt.symbol_type in symbol_types] \
                    + [gt for (_, gt) in gt_symbols if gt.symbol_type in symbol_types]

                tp, fp, fn = tuple(list(map(len, (l_tp, l_fp, l_fn))))

                try:
                    return (tp, fp, fn), precision_recall_f1(tp, fp, fn), (l_tp, l_fp, l_fn)
                except ZeroDivisionError:
                    return (tp, fp, fn), None, (l_tp, l_fp, l_fn)

            def note_sub_group(lists, prf2metric: PRF2Metrics):
                l_tp, _, _ = lists
                if prf2metric == PRF2Metrics.NOTE_ALL:
                    l_true = [(p, gt) for p, gt in l_tp if p.graphical_connection == gt.graphical_connection and p.neume_start == gt.neume_start]
                    l_false = [(p, gt) for p, gt in l_tp if p.neume_start != gt.neume_start or gt.graphical_connection != p.graphical_connection]
                elif prf2metric == PRF2Metrics.NOTE_GC:
                    l_true = [(p, gt) for p, gt in l_tp if p.graphical_connection == gt.graphical_connection]
                    l_false = [(p, gt) for p, gt in l_tp if p.graphical_connection != gt.graphical_connection]
                elif prf2metric == PRF2Metrics.NOTE_NS:
                    l_true = [(p, gt) for p, gt in l_tp if p.neume_start == gt.neume_start]
                    l_false = [(p, gt) for p, gt in l_tp if p.neume_start != gt.neume_start]
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
                l_true = [(p, gt) for p, gt in l_tp if gt.accidental == p.accidental]
                l_false = [(p, gt) for p, gt in l_tp if gt.accidental != p.accidental]
                true, false = tuple(list(map(len, (l_true, l_false))))
                try:
                    return (true, false, true + false), true / (false + true), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, true + false), None, (l_true, l_false, [])

            all_counts, all_metrics, all_ = sub_group([SymbolType.NOTE_COMPONENT, SymbolType.ACCID, SymbolType.CLEF])
            note_counts, note_metrics, notes = sub_group([SymbolType.NOTE_COMPONENT])
            clef_counts, clef_metrics, clefs = sub_group([SymbolType.CLEF])
            accid_counts, accid_metrics, accids = sub_group([SymbolType.ACCID])

            note_all_counts, note_all_metrics, note_all = note_sub_group(notes, PRF2Metrics.NOTE_ALL)
            note_gc_counts, note_gc_metrics, note_gc = note_sub_group(notes, PRF2Metrics.NOTE_GC)
            note_ns_counts, note_ns_metrics, note_ns = note_sub_group(notes, PRF2Metrics.NOTE_NS)
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
                note_gc_counts,
                note_ns_counts,
                note_pis_counts,
                clef_all_counts,
                clef_pis_counts,
                accid_all_counts,
                (sequence_ed[1], sequence_ed[0], sum(sequence_ed)),
                (sequence_ed_nc[1], sequence_ed_nc[0], sum(sequence_ed_nc)),
            ]]), axis=0)

            acc_metrics = np.concatenate((acc_metrics, [[
                note_all_metrics,
                note_gc_metrics,
                note_ns_metrics,
                note_pis_metrics,
                clef_all_metrics,
                clef_pis_metrics,
            ]]), axis=0)

        acc_counts = acc_counts.sum(axis=0)
        acc_acc = (acc_counts[:, AccCounts.TRUE] / acc_counts[:, AccCounts.TOTAL]).reshape((-1, 1))
        return f_metrics.mean(axis=0), counts.sum(axis=0), acc_counts, acc_acc


if __name__ == '__main__':
    from omr.symboldetection.predictor import SymbolDetectionPredictor, create_predictor, PredictorTypes, SymbolDetectionPredictorParameters
    from prettytable import PrettyTable
    from database import DatabaseBook
    b = DatabaseBook('Graduel')
    eval_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[12:13]]
    print([e.page.location.local_path() for e in eval_pcgts])
    predictor = create_predictor(
        PredictorTypes.PIXEL_CLASSIFIER,
        SymbolDetectionPredictorParameters([b.local_path(os.path.join('pc_symbol_detection', 'model'))]))
    gt_symbols, pred_symbols = [], []
    for p in predictor.predict(eval_pcgts):
        pred_symbols.append(p.symbols)
        gt_symbols.append(p.line.operation.music_line.symbols)

    evaluator = SymbolDetectionEvaluator()
    metrics, counts, acc_counts, acc_acc = evaluator.evaluate(gt_symbols, pred_symbols)

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


    at = PrettyTable()
    at.add_column("Type", ["Note all", "Note GC", "Note NS", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
    at.add_column("True", acc_counts[:, AccCounts.TRUE])
    at.add_column("False", acc_counts[:, AccCounts.FALSE])
    at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
    at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

    print(at.get_string())

