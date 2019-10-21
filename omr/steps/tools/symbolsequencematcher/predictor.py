from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage, DatabaseBook
from typing import List, Optional, NamedTuple

from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step

from omr.steps.tools.symbolsequencematcher.meta import Meta
from omr.steps.symboldetection.predictor import PredictionResult as SymbolsPredictionResult
from database.file_formats.pcgts.page.musicsymbol import SymbolType
from database.file_formats.pcgts.page.definitions import MusicSymbolPositionInStaff

from database.file_formats.pcgts import MusicSymbol

from enum import Enum

from database import database_page
from omr.steps.tools.symbolsequencematcher.construct import Constructor, SymbolInsertion


class Codec(Enum):
    PREDICTION = 1
    PCGTS = 2
    LINE = 3


def codec(symbol_prediction, codec: Codec = Codec.PREDICTION):
    sequence = []
    if codec == codec.PREDICTION:
        predicted_music_symbols = [symbol for page in symbol_prediction for music_line in page.music_lines
                                   for symbol in music_line.symbols if
                                   symbol.symbol_type == symbol.symbol_type.NOTE]
        for symbol_ind, symbol in enumerate(predicted_music_symbols):
            diff = 0 if symbol_ind == 0 else symbol.position_in_staff - predicted_music_symbols[
                symbol_ind - 1].position_in_staff
            sequence.append(diff)
    if codec == codec.PCGTS:
        gt_symbols = [symbol for page in symbol_prediction for mb in page.pcgts().page.music_blocks() for ml in mb.lines
                      for symbol in ml.symbols if symbol.symbol_type == symbol.symbol_type.NOTE]

        def base_c(symbol: MusicSymbol):
            octave = symbol.octave
            if symbol.note_name < 2:
                octave += 1
            return octave

        for symbol_ind, symbol in enumerate(gt_symbols):
            diff_symbol = 0 if symbol_ind == 0 else symbol.note_name - gt_symbols[
                symbol_ind - 1].note_name
            diff_octave = 0 if symbol_ind == 0 else base_c(symbol) - base_c(gt_symbols[
                                                                                symbol_ind - 1])

            diff = diff_symbol + diff_octave * 7
            sequence.append(diff)

    if codec == codec.LINE:
        predicted_music_symbols = [symbol for symbol in symbol_prediction if
                                   symbol.symbol_type == symbol.symbol_type.NOTE]

        def base_c(symbol: MusicSymbol):
            octave = symbol.octave
            if symbol.note_name < 2:
                octave += 1
            return octave

        for symbol_ind, symbol in enumerate(predicted_music_symbols):
            diff_symbol = 0 if symbol_ind == 0 else symbol.note_name - predicted_music_symbols[
                symbol_ind - 1].note_name
            diff_octave = 0 if symbol_ind == 0 else base_c(symbol) - base_c(predicted_music_symbols[
                                                                                symbol_ind - 1])

            diff = diff_symbol + diff_octave * 7
            sequence.append(diff)

    return sequence


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        self.symbol_detection = Step.create_predictor(AlgorithmTypes.SYMBOLS_PC, settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None, gt=None) -> AlgorithmPredictionResultGenerator:
        predctions: List[SymbolsPredictionResult] = list(self.symbol_detection.predict(pages, callback))
        aligner = Aligner(predctions, gt)

        opt_codes = aligner.get_lev_optcodes(CalcType.DELTAPIS)

        #for x in opt_codes:
        #    print(x)
        clefs = aligner.calculate_clef_positions()
        from omr.steps.tools.symbolsequencematcher.construct import Constructor
        for page in predctions:
            for line in page.music_lines:
                line.line.operation.music_line.symbols = line.symbols

            page.pcgts.page.annotations.connections.clear()
            page.pcgts.to_file(page.dataset_page.file('pcgts').local_path())
        aligner_name1 = Aligner(predctions, gt)
        opt_codes_name1 = aligner_name1.get_lev_optcodes(CalcType.DELTANNAME)

        constructor = Constructor(prediction=predctions)
        modified_prediction = constructor.rebuild_prediction(clefs)
        for page in modified_prediction:
            for line in page.music_lines:
                line.line.operation.music_line.symbols = line.symbols

            page.pcgts.page.annotations.connections.clear()
            page.pcgts.to_file(page.dataset_page.file('pcgts').local_path())

        aligner_name = Aligner(modified_prediction, gt)
        opt_codes_name = aligner_name.get_lev_optcodes(CalcType.DELTANNAME)


class CalcType(Enum):
    DELTAPIS = 0
    DELTANNAME = 1


class Aligner:
    from database.file_formats.pcgts import Page

    def __init__(self, prediction, gt_pcgts: [database_page]):
        self.prediction: List[SymbolsPredictionResult] = prediction
        self.gt_pcgts: [database_page] = gt_pcgts
        self.opt_codes = None

    def get_lev_optcodes(self, type : CalcType = CalcType.DELTAPIS):
        if self.opt_codes is not None:
            return self.opt_codes
        import edit_distance

        prediction_sequence = []
        gt_sequence = []
        ### generate sequences to compare
        if type == CalcType.DELTAPIS:
            prediction_sequence = codec(self.prediction, Codec.PREDICTION)
            gt_sequence = codec(gt_pcgts, Codec.PCGTS)

        if type == CalcType.DELTANNAME:
            for page in self.prediction:
                for line in page.music_lines:
                    line.line.operation.music_region.update_note_names()

            prediction_sequence = [symbol.note_name for page in self.prediction for music_line in page.music_lines
                                   for symbol in music_line.line.operation.music_line.symbols if symbol.symbol_type == symbol.symbol_type.NOTE]

            gt_sequence = [symbol.note_name for page in self.gt_pcgts for mb in page.pcgts().page.music_blocks() for ml
                           in mb.lines for symbol in ml.symbols if symbol.symbol_type == symbol.symbol_type.NOTE]

        seq = edit_distance.SequenceMatcher()
        seq.set_seqs(prediction_sequence, gt_sequence)

        length = len(seq.get_opcodes())
        equals = len([1 for x in seq.get_opcodes() if x[0] == "equal"])
        print('Length {}, Equals {}, % {}'.format(length, equals, equals / length))
        print(seq.ratio())
        self.opt_codes = seq.get_opcodes()
        return self.opt_codes

    def calculate_clef_positions(self) -> List[SymbolInsertion]:

        gt_symbols = [symbol for page in self.gt_pcgts for mb in page.pcgts().page.music_blocks() for ml in mb.lines
                      for symbol in ml.symbols]
        from database.file_formats.pcgts.page.line import Line
        gt_symbols_note = [symbol for symbol in gt_symbols if symbol.symbol_type == symbol.symbol_type.NOTE]

        dummy_line = [symbol for page in self.prediction for music_line in page.music_lines
                      for symbol in music_line.symbols]

        iters = iter(self.get_lev_optcodes(CalcType.DELTAPIS))
        opt_counter = -1

        last_change_position_in_line = -1
        last_clef_position_on_staff = None

        last_equal = -1

        predicted_clef = None
        clefs_to_insert = []
        for s_ind, s in enumerate(dummy_line):
            if s.symbol_type == s.symbol_type.CLEF:
                predicted_clef = s
            if s.symbol_type == s.symbol_type.NOTE:
                opt_counter += 1
                current_optcode = next(iters)

                while current_optcode[0] == 'insert':
                    matched_gt_symbol = gt_symbols_note[current_optcode[3]:current_optcode[4]][0]
                    current_optcode = next(iters)
                    opt_counter += 1

                if current_optcode[0] == "equal":
                    matched_gt_symbol = gt_symbols_note[current_optcode[3]:current_optcode[4]][0]
                    calculated_clef_symbol = s.pis_octave(matched_gt_symbol.note_name)

                    def get_best_clef_match(clef_tuples, predicted_clef):
                        from database.file_formats.pcgts.page.musicsymbol import ClefType, MusicSymbol, SymbolType
                        possible_c_clef = [clef_tuples[0][0], clef_tuples[1][0]]
                        possible_f_clef = [clef_tuples[0][1], clef_tuples[1][1]]
                        if predicted_clef in possible_c_clef:
                            return predicted_clef, ClefType.C
                        if predicted_clef in possible_f_clef:
                            return predicted_clef, ClefType.F

                        if possible_c_clef[0] <= 0:
                            return possible_c_clef[1], ClefType.C
                        else:
                            return possible_c_clef[0], ClefType.C

                    best_clef_pis = get_best_clef_match(calculated_clef_symbol, predicted_clef)

                    if last_clef_position_on_staff is None:
                        last_clef_position_on_staff = best_clef_pis
                        last_change_position_in_line = 0

                    if best_clef_pis != last_clef_position_on_staff:
                        clefs_to_insert.append(SymbolInsertion(PISequence=last_change_position_in_line,
                                                               SymbolType=SymbolType.CLEF,
                                                               PIS=MusicSymbolPositionInStaff(last_clef_position_on_staff[0]),
                                                               NoteType=last_clef_position_on_staff[1]))
                        last_change_position_in_line = s_ind
                        last_clef_position_on_staff = best_clef_pis

                    if s_ind == len(dummy_line) -1:
                        clefs_to_insert.append(SymbolInsertion(PISequence=last_change_position_in_line,
                                                               SymbolType=SymbolType.CLEF,
                                                               PIS=MusicSymbolPositionInStaff(last_clef_position_on_staff[0]),
                                                               NoteType=last_clef_position_on_staff[1]))

        def prune_possible_clefs(clefs_to_insert: List[SymbolInsertion], min_seqeuence_length=4):
            while True:
                con = False

                for i in range(1, len(clefs_to_insert) - 1, 1):
                    previous_clef = clefs_to_insert[i - 1]
                    current_clef = clefs_to_insert[i]
                    next_clef = clefs_to_insert[i + 1]
                    dist = next_clef.PISequence - current_clef.PISequence
                    if previous_clef.PIS == next_clef.PIS and dist <= min_seqeuence_length * 2:
                        con = True
                        del clefs_to_insert[i]
                        break
                if not con:
                    break

            for r in range(1, min_seqeuence_length + 1, 1):
                while True:
                    con = False
                    for i in range(len(clefs_to_insert) - 1):
                        previous_clef = None
                        if i != 0:
                            previous_clef = clefs_to_insert[i - 1]
                        current_clef = clefs_to_insert[i]
                        next_clef = clefs_to_insert[i + 1]
                        dist = next_clef.PISequence - current_clef.PISequence
                        if dist <= r:
                            con = True
                            del clefs_to_insert[i]
                            break
                    if not con:
                        break

            while True:
                con = False

                for i in range(0, len(clefs_to_insert) - 1, 1):
                    current_clef = clefs_to_insert[i]
                    next_clef = clefs_to_insert[i + 1]
                    if current_clef.PIS == next_clef.PIS:
                        con = True
                        del clefs_to_insert[i + 1]
                        break
                if not con:
                    break

            return clefs_to_insert
        gt_clef = [(y.position_in_staff.value, y.clef_type.name, x) for x, y in enumerate(gt_symbols) if
                   y.symbol_type == y.symbol_type.CLEF]
        clefs_to_insert = prune_possible_clefs(clefs_to_insert, 5)

        #print(gt_clef)
        #print([(x.PIS.value, x.NoteType.value, x.PISequence) for x in clefs_to_insert])

        return clefs_to_insert


if __name__ == '__main__':
    from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
    import os

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    import django

    django.setup()
    #DatabaseBook.create()
    gt_book = DatabaseBook('Graduel')
    gt_pcgts = gt_book.pages()[23:28]
    pred_book = DatabaseBook('Graduel_prediction')
    pred_pcgts = pred_book.pages()[23:28]
    for pcgts, gpcgts in zip(pred_pcgts, gt_pcgts):
        pcgts.pcgts().page.sort_blocks()
        gpcgts.pcgts().page.sort_blocks()

    meta = Step.meta(AlgorithmTypes.SYMBOLS_PC)
    model = meta.best_model_for_book(pred_book)
    settings = AlgorithmPredictorSettings(model=model)
    predictor = Predictor(settings)
    predictor.predict(pred_pcgts, gt=gt_pcgts)
    print(pred_pcgts[0].page)
