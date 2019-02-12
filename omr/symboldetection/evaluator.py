from omr.symboldetection.predictor import SymbolDetectionPredictor, create_predictor, PredictorTypes, PredictorParameters
from omr.datatypes import PcGts, Symbol, Clef, Neume, NoteComponent, Accidental, Coords, Point, SymbolType, GraphicalConnectionType, ClefType
from typing import List, Tuple
import os
import numpy as np
from enum import IntEnum

class PRF2Metrics(IntEnum):
    SYMBOL = 0
    NOTE = 1
    NOTE_ALL = 2
    NOTE_GC = 3
    NOTE_NS = 5
    CLEF = 6
    CLEF_ALL = 7
    ACCID = 9

    COUNT = 10


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


class SymbolDetectionEvaluator:
    def __init__(self, predictor: SymbolDetectionPredictor):
        self.predictor = predictor

    def evaluate(self, pcgts_files: List[PcGts], min_distance=5):
        def precision_recall_f1(tp, fp, fn) -> Tuple[float, float, float]:
            return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)

        min_distance_sqr = min_distance ** 2
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
        acc_metrics = np.zeros((0, 4), dtype=float)
        counts = np.zeros((0, 9, Counts.COUNT), dtype=int)

        for p in self.predictor.predict(pcgts_files):
            pairs = []
            p_symbols = extract_coords_of_symbols(p.symbols)
            gt_symbols_orig = extract_coords_of_symbols(p.line.operation.music_line.symbols)
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
            precision, recall, f1 = precision_recall_f1(n_tp, n_fp, n_fn)

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
                l_tp, l_fp, l_fn = lists
                if prf2metric == PRF2Metrics.NOTE_ALL:
                    l_true = [(p, gt) for p, gt in l_tp if p.graphical_connection == gt.graphical_connection and p.neume_start == gt.neume_start]
                    l_false = [(p, gt) for p, gt in l_tp if p.neume_start != gt.neume_start or gt.graphical_connection != p.graphical_connection]
                elif prf2metric == PRF2Metrics.NOTE_GC:
                    l_true = [(p, gt) for p, gt in l_tp if p.graphical_connection == gt.graphical_connection]
                    l_false = [(p, gt) for p, gt in l_tp if p.graphical_connection != gt.graphical_connection]
                elif prf2metric == PRF2Metrics.NOTE_NS:
                    l_true = [(p, gt) for p, gt in l_tp if p.neume_start == gt.neume_start]
                    l_false = [(p, gt) for p, gt in l_tp if p.neume_start != gt.neume_start]
                try:
                    true, false = tuple(list(map(len, (l_true, l_false))))
                    return (true, false, 0), true / (true + false), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, 0), None, (l_true, l_false, [])

            def clef_sub_group(lists, prf2metric: PRF2Metrics):
                l_tp, l_fp, l_fn = lists
                l_true = [(p, gt) for p, gt in l_tp if gt.clef_type == p.clef_type]
                l_false = [(p, gt) for p, gt in l_tp if gt.clef_type != p.clef_type]
                try:
                    true, false = tuple(list(map(len, (l_true, l_false))))
                    return (true, false, 0), true / (false + true), (l_true, l_false, [])
                except ZeroDivisionError:
                    return (true, false, 0), None, (l_true, l_false, [])

            all_counts, all_metrics, all_ = sub_group([SymbolType.NOTE_COMPONENT, SymbolType.ACCID, SymbolType.CLEF])
            note_counts, note_metrics, notes = sub_group([SymbolType.NOTE_COMPONENT])
            clef_counts, clef_metrics, clefs = sub_group([SymbolType.CLEF])
            accis_counts, accid_metrics, accids = sub_group([SymbolType.ACCID])

            note_all_counts, note_all_metrics, note_all = note_sub_group(notes, PRF2Metrics.NOTE_ALL)
            note_gc_counts, note_gc_metrics, note_gc = note_sub_group(notes, PRF2Metrics.NOTE_GC)
            note_ns_counts, note_ns_metrics, note_ns = note_sub_group(notes, PRF2Metrics.NOTE_NS)

            clef_all_counts, clef_all_metrics, clefs_all = clef_sub_group(clefs, PRF2Metrics.CLEF_ALL)

            counts = np.concatenate((counts, [[[n_tp, n_fp, n_fn],
                                               all_counts,
                                               note_counts,
                                               note_all_counts,
                                               note_gc_counts,
                                               note_ns_counts,
                                               clef_counts,
                                               clef_all_counts,
                                               accis_counts]]), axis=0)

            acc_metrics = np.concatenate((acc_metrics, [[
                note_all_metrics,
                note_gc_metrics,
                note_ns_metrics,
                clef_all_metrics,
            ]]), axis=0)

        total_counts = counts.sum(axis=0)
        out_counts = np.array([total_counts[3], total_counts[4], total_counts[5], total_counts[7]])
        return f_metrics.mean(axis=0), counts.sum(axis=0), out_counts[:, 0] / out_counts.sum(axis=1)


if __name__ == '__main__':
    import main.book as book
    b = book.Book('Graduel')
    eval_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:1] + b.pages()[4:10]]
    pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER,
                            PredictorParameters([b.local_path(os.path.join('pc_symbol_detection', 'model'))]))
    evaluator = SymbolDetectionEvaluator(pred)
    metrics, counts, acc_metrics = evaluator.evaluate(eval_pcgts)

    print(metrics)
    print(counts)
    print(acc_metrics)
    # print("{} symbol true positives composed of {} nc, clef, accid".format(np.sum(single_counts[0]), single_counts[0]))
    # print("{} symbol false positives composed of {} nc, clef, accid".format(np.sum(single_counts[1]), single_counts[1]))
    # print("{} symbol false negatives composed of {} nc, clef, accid".format(np.sum(single_counts[2]), single_counts[2]))
    # print("{} gt symbols composed of {} nc, clef, accid".format(np.sum(single_counts[3]), single_counts[3]))

