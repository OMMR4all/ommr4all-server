from omr.symboldetection.predictor import SymbolDetectionPredictor, create_predictor, PredictorTypes, PredictorParameters
from omr.datatypes import PcGts, Symbol, Clef, Neume, NoteComponent, Accidental, Coords, Point, SymbolType
from typing import List, Tuple
import os
import numpy as np


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

        metrics = np.zeros((0, 3, 3), dtype=float)
        counts = np.zeros((0, 3, 3), dtype=int)
        single_counts = np.zeros((0, 4, 3), dtype=int)

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

            type_tp = len([p for p in pairs if p[0][1].symbol_type == p[1][1].symbol_type])
            type_accuracy = type_tp / len(pairs)
            type_prf = precision_recall_f1(type_tp, n_fp + len(pairs) - type_tp, n_fn)

            nc_pairs: List[Tuple[NoteComponent, NoteComponent]] \
                = [(p, gt) for (_, p), (_, gt) in pairs if p.symbol_type == gt.symbol_type and p.symbol_type == SymbolType.NOTE_COMPONENT]
            clef_pairs: List[Tuple[Clef, Clef]] = [(p, gt) for (_, p), (_, gt) in pairs if p.symbol_type == gt.symbol_type and p.symbol_type == SymbolType.CLEF]
            accid_pairs: List[Tuple[Accidental, Accidental]] = [(p, gt) for (_, p), (_, gt) in pairs if p.symbol_type == gt.symbol_type and p.symbol_type == SymbolType.ACCID]
            wrong_pairs = [(p, gt) for (_, p), (_, gt) in pairs if p.symbol_type != gt.symbol_type]

            nc_tp = len([p for p, gt in nc_pairs if p.note_type == gt.note_type and p.graphical_connection == gt.graphical_connection and p.neume_start == gt.neume_start])
            wrong_nc_symbols = [(p, gt) for p, gt in nc_pairs if p.note_type != gt.note_type or p.graphical_connection != gt.graphical_connection or p.neume_start != gt.neume_start]
            clef_tp = len([p for p, gt in clef_pairs if p.clef_type == gt.clef_type])
            wrong_clef_symbols = [(p, gt) for p, gt in clef_pairs if p.clef_type != gt.clef_type]
            accid_tp = len([p for p, gt in accid_pairs if p.accidental == gt.accidental])
            wrong_accid_symbols = [(p, gt) for p, gt in accid_pairs if p.accidental != gt.accidental]

            n_nc = len([c for c, s in gt_symbols_orig if s.symbol_type == SymbolType.NOTE_COMPONENT])
            n_clef = len([c for c, s in gt_symbols_orig if s.symbol_type == SymbolType.CLEF])
            n_accid = len([c for c, s in gt_symbols_orig if s.symbol_type == SymbolType.ACCID])

            n_fp_nc = len([c for c, s in p_symbols if s.symbol_type == SymbolType.NOTE_COMPONENT])
            n_fp_clef = len([c for c, s in p_symbols if s.symbol_type == SymbolType.CLEF])
            n_fp_accid = len([c for c, s in p_symbols if s.symbol_type == SymbolType.ACCID])

            all_fn_symbols = gt_symbols + wrong_pairs + wrong_nc_symbols + wrong_clef_symbols + wrong_accid_symbols
            n_fn_nc = len([c for c, s in all_fn_symbols if s.symbol_type == SymbolType.NOTE_COMPONENT])
            n_fn_clef = len([c for c, s in all_fn_symbols if s.symbol_type == SymbolType.CLEF])
            n_fn_accid = len([c for c, s in all_fn_symbols if s.symbol_type == SymbolType.ACCID])

            tp = nc_tp + clef_tp + accid_tp
            fp = n_fp
            fn = n_fn + (n_tp - tp)

            full_prf = precision_recall_f1(tp, fp, fn)

            metrics = np.concatenate((metrics, [[[precision, recall, f1], type_prf, full_prf]]), axis=0)
            counts = np.concatenate((counts, [[[n_tp, n_fp, n_fn], [type_tp, n_fp + len(pairs) - type_tp, n_fn], [tp, fp, fn]]]), axis=0)
            single_counts = np.concatenate((single_counts, [[(nc_tp, clef_tp, accid_tp),
                                                             (n_fp_nc, n_fp_clef, n_fp_accid),
                                                             (n_fn_nc, n_fn_clef, n_fn_accid),
                                                             (n_nc, n_clef, n_accid),]]), axis=0)

        return metrics.mean(axis=0), counts.sum(axis=0), single_counts.sum(axis=0)






if __name__ == '__main__':
    import main.book as book
    b = book.Book('Graduel')
    eval_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:1] + b.pages()[4:5]]
    pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER,
                            PredictorParameters([b.local_path(os.path.join('pc_symbol_detection', 'model'))]))
    evaluator = SymbolDetectionEvaluator(pred)
    metrics, counts, single_counts = evaluator.evaluate(eval_pcgts)

    print(metrics)
    print(counts)
    print("{} symbol true positives composed of {} nc, clef, accid".format(np.sum(single_counts[0]), single_counts[0]))
    print("{} symbol false positives composed of {} nc, clef, accid".format(np.sum(single_counts[1]), single_counts[1]))
    print("{} symbol false negatives composed of {} nc, clef, accid".format(np.sum(single_counts[2]), single_counts[2]))
    print("{} gt symbols composed of {} nc, clef, accid".format(np.sum(single_counts[3]), single_counts[3]))

