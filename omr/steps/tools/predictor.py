from difflib import SequenceMatcher

import numpy as np

from database import DatabaseBook, DatabasePage
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.book.document import Document
from database.file_formats.importer.mondodi.simple_import import simple_monodi_data_importer, \
    simple_monodi_data_importer2, Neume
from database.file_formats.pcgts import *
import logging
from typing import List, Optional, Tuple, NamedTuple

from database.file_formats.pcgts.page import Annotations, SyllableConnector, Sentence, Syllable
from database.file_formats.pcgts.page.annotations import Connection
from database.model import Model, MetaId
from omr.dataset import RegionLineMaskData
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.layout.correction_tools.connectedcomponentsselector.predictor import ResultMeta
from omr.steps.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
from omr.steps.preprocessing.meta import Meta
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings, \
    AlgorithmPredictorParams, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator
import multiprocessing

from omr.steps.preprocessing.preprocessing import Preprocessing
from omr.steps.step import Step
from omr.steps.symboldetection.predictor import PredictionResult

logger = logging.getLogger(__name__)


class SingleLinePredictionResult(NamedTuple):
    symbols: List[MusicSymbol]
    line: RegionLineMaskData
    text_line: Optional[Line]
    text: List[str]
    annontations: Annotations
    syllables: List[Syllable]

    def to_dict(self):
        return {'symbols': [s.to_json() for s in self.symbols],
                'id': self.line.operation.music_line.id}


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    lines: List[SingleLinePredictionResult]

    def to_dict(self):
        return {}

    def store_to_page(self):
        pages = set(x.line.operation.page.p_id for x in self.lines)
        for i in pages:
            lines_of_page = [x for x in self.lines if i == x.line.operation.page.p_id]
            page = lines_of_page[0].line.operation.page
            pcgts = lines_of_page[0].line.operation.pcgts
            pcgts.page.annotations.connections.clear()
            connections_of_page: List[Connection] = []
            for line in lines_of_page:
                line.line.operation.music_line.symbols = line.symbols
                line.text_line.sentence = Sentence(syllables=line.syllables)

                connections_of_page += line.annontations.connections
            if len(lines_of_page) > 0:
                annotation_cp = lines_of_page[0].annontations
                annotation = Annotations(annotation_cp.page, connections_of_page)
                pcgts.page.annotations = annotation

            pcgts.to_file(page.location.file('pcgts').local_path())
        pass


class PreprocessingPredictor(AlgorithmPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        self.document_text = settings.params.documentText

        meta = Step.meta(AlgorithmTypes.SYMBOLS_PC)
        # from ommr4all.settings import BASE_DIR
        # model = Model(
        #    MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/fraktur/text_calamari/', meta.type()))
        # settings = AlgorithmPredictorSettings(
        #    model=model,
        # )
        self.symbol_predictor = meta.create_predictor(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:

        import json
        # print(json.loads(self.settings.params.documentText))
        json_string = json.loads(self.settings.params.documentText)
        abc = simple_monodi_data_importer2(json_string, ignore_liquescent=True)
        # print(abc)
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(self.document_id)
        pages = [DatabasePage(book, x) for x in document.pages_names]
        # print(document.start.line_id)
        line_symbols = list(self.symbol_predictor.predict(pages))
        start = False
        index = 0
        stop = False
        result: List[SingleLinePredictionResult] = []

        for i in line_symbols:
            if stop:
                break
            i: PredictionResult = i

            # print("123123")
            # print(i.music_lines)
            for ind, t in enumerate(sorted(i.music_lines, key=lambda x: x.line.operation.music_line.center_y())):

                # print(document.start.line_id)
                # print(t.line.operation.music_line.id)
                text_line = t.line.operation.page.closest_below_text_line_to_music_line(t.line.operation.music_line,
                                                                                        True)
                if (
                        text_line.id == document.end.line_id and t.line.operation.page.p_id == document.end.page_id) or index == len(
                    abc):
                    stop = True
                    break
                # text_line = t.line.operation.page.line_by_id(document.start.line_id)
                music_line = t.line.operation.page.closest_music_line_to_text_line(text_line)
                # print(ind)
                pred = "".join([str(s.note_name) for s in t.symbols if s.symbol_type == s.symbol_type.NOTE])
                pred_symbols = [s for s in t.symbols if s.symbol_type == s.symbol_type.NOTE]
                # print(pred)
                if (
                        text_line.id == document.start.line_id and t.line.operation.page.p_id == document.start.page_id) or start:
                    gt = "".join([str(x.note_name) for s in abc[index].neumes for x in s.symbols])
                    # print(gt)
                    gt_symbols = [x for s in abc[index].neumes for x in s.symbols]
                    gt_text = [s.syllable.text for s in abc[index].neumes]
                    gt_syllables = [s.syllable for s in abc[index].neumes]
                    neumes_of_line = abc[index].neumes
                    sm = SequenceMatcher(a=pred, b=gt, autojunk=False, isjunk=False)
                    # text_line.update_note_names()
                    gt_aligned_symbols = []

                    matches = []
                    # print(pred)
                    # print(gt)
                    for opcode, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes():
                        if opcode == 'equal':
                            # print("equal")
                            # print(gt_symbols[gt_start:gt_end])
                            # gt_aligned_symbols += pred_symbols[pred_start:pred_end]
                            pred_symbols_part = pred_symbols[pred_start:pred_end]
                            gt_symbols_part = gt_symbols[gt_start:gt_end]
                            for pred_symbol, gt_symbol in zip(pred_symbols_part, gt_symbols_part):
                                gt_symbol: MusicSymbol = gt_symbol
                                pred_symbol: MusicSymbol = pred_symbol
                                gt_symbol.coord = pred_symbol.coord
                                gt_symbol.position_in_staff = pred_symbol.position_in_staff

                            gt_aligned_symbols += gt_symbols_part

                        elif opcode == 'insert':
                            # print("insert")
                            # print(gt_symbols[gt_start:gt_end])
                            # symbol = gt_symbols[gt_start:gt_end][0]
                            gt_aligned_symbols += gt_symbols[gt_start:gt_end]
                        elif opcode == 'delete':
                            pass
                        elif opcode == 'replace':
                            # print("replace")
                            # print(gt_symbols[gt_start:gt_end])
                            gt_aligned_symbols += gt_symbols[gt_start:gt_end]
                    pred2 = "".join(
                        [str(s.note_name) for s in gt_aligned_symbols if s.symbol_type == s.symbol_type.NOTE])

                    def get_staff_difference(n1: MusicSymbol, n2: MusicSymbol):
                        n1_octave = n1.octave
                        n2_octave = n2.octave
                        n1_note_name = n1.note_name
                        n2_note_name = n2.note_name
                        difference = (n1_note_name.octave_value() + 7 * n1_octave) - (
                                n2_note_name.octave_value() + 7 * n2_octave)
                        return difference

                    def get_best_point(image, scaled_x_pos_prev, left=0, kernelsize=10):
                        if len(image.shape) == 2 and image.shape[1] > 0:

                            kernel_row = [1 for x in range(kernelsize)]
                            k = np.array([kernel_row for x in range(kernelsize)])
                            from scipy import ndimage
                            weight_image = ndimage.convolve(image, k, mode='constant', cval=0.0)
                            result = np.where(weight_image == np.amax(weight_image))
                            return int(np.mean(result[1])) + left + scaled_x_pos_prev
                        else:
                            return scaled_x_pos_prev + left

                    def correct_symbols_sequence(aligned_gt: List[MusicSymbol], pred: List[MusicSymbol],
                                                 mask: SingleLinePredictionResult):
                        binary = i.dataset_page.file('binary_highres_preproc').local_path()
                        from PIL import Image
                        import numpy as np
                        binary = np.array(Image.open(binary))
                        # print(type(binary))
                        # grayscale =
                        # page.file('gray_highres_preproc').local_path()
                        # image = t.line.operation.page.file
                        stafflines = t.line.operation.music_line.staff_lines
                        aabb = stafflines.aabb().to_coords()
                        coord = mask.line.operation.page.page_to_image_scale(aabb, PageScaleReference.HIGHRES)
                        left = int(coord.aabb().left())
                        # print(coord)
                        # print(type(coord))
                        # print(left)
                        sub_image = coord.extract_from_image(binary)
                        # from matplotlib import pyplot as plt
                        # plt.imshow(sub_image)
                        # plt.show()

                        repeat = True
                        while repeat:
                            repeat = False
                            for index, symbol in enumerate(aligned_gt):
                                if symbol.position_in_staff == MusicSymbolPositionInStaff.UNDEFINED:
                                    # todo octave
                                    if aligned_gt[index - 1].position_in_staff != MusicSymbolPositionInStaff.UNDEFINED:
                                        difference_prev = get_staff_difference(aligned_gt[index - 1], symbol)
                                        print("prevS: NN {}, ov: {}, octave {}".format(aligned_gt[index - 1].note_name,
                                                                                       aligned_gt[
                                                                                           index - 1].note_name.octave_value(),
                                                                                       aligned_gt[index - 1].octave))
                                        print("S: NN {}, ov: {}, octave {}".format(symbol.note_name,
                                                                                   symbol.note_name.octave_value(),
                                                                                   symbol.octave))
                                        print(difference_prev)

                                        symbol.position_in_staff = MusicSymbolPositionInStaff(
                                            aligned_gt[index - 1].position_in_staff - difference_prev if 0 <=
                                                                                                         aligned_gt[
                                                                                                             index - 1].position_in_staff - difference_prev <= 14 else 0)
                                        if len(aligned_gt) > index + 1 and aligned_gt[
                                            index + 1].position_in_staff != MusicSymbolPositionInStaff.UNDEFINED:
                                            x_pos_prev = aligned_gt[index - 1].coord.x
                                            x_pos_after = aligned_gt[index + 1].coord.x
                                            scaled_x_pos_prev = int(
                                                mask.line.operation.page.page_to_image_scale(x_pos_prev,
                                                                                             PageScaleReference.HIGHRES)) - left
                                            scaled_x_pos_after = int(
                                                mask.line.operation.page.page_to_image_scale(x_pos_after,
                                                                                             PageScaleReference.HIGHRES)) - left

                                            # print(scaled_x_pos_prev)
                                            # print(scaled_x_pos_after)
                                            sub_sub_image = sub_image[:, scaled_x_pos_prev: scaled_x_pos_after]

                                            coord = get_best_point(sub_sub_image, scaled_x_pos_prev, left)
                                            # plt.imshow(sub_sub_image)
                                            # plt.show()
                                            print(coord)
                                            x_pos = mask.line.operation.page.image_to_page_scale(coord,
                                                                                                 PageScaleReference.HIGHRES)
                                        else:
                                            x_pos = mask.line.operation.page.image_to_page_scale(int(
                                                mask.line.operation.page.page_to_image_scale(aligned_gt[index - 1].coord.x,
                                                                                             PageScaleReference.HIGHRES)) + 10,
                                                                                                 PageScaleReference.HIGHRES)
                                        symbol.coord = t.line.operation.music_line.staff_lines.compute_coord_by_position_in_staff(
                                            x_pos, symbol.position_in_staff)

                                    elif aligned_gt[
                                        index + 1].position_in_staff != MusicSymbolPositionInStaff.UNDEFINED:
                                        difference_after = get_staff_difference(aligned_gt[index + 1], symbol)
                                        print("prevS: NN {}, ov: {}, octave {}".format(aligned_gt[index + 1].note_name,
                                                                                       aligned_gt[
                                                                                           index + 1].note_name.octave_value(),
                                                                                       aligned_gt[index + 1].octave))
                                        print("S: NN {}, ov: {}, octave {}".format(symbol.note_name,
                                                                                   symbol.note_name.octave_value(),
                                                                                   symbol.octave))
                                        print(difference_after)

                                        # difference_after = aligned_gt[index + 1].octave * 8 + aligned_gt[
                                        #    index + 1].note_name - (symbol.octave * 8 + symbol.note_name)
                                        symbol.position_in_staff = MusicSymbolPositionInStaff(
                                            aligned_gt[index + 1].position_in_staff - difference_after if 0 <=
                                                                                                          aligned_gt[
                                                                                                              index + 1].position_in_staff - difference_after <= 14 else 0)

                                        x_pos_prev = aligned_gt[index - 1].coord.x
                                        x_pos_after = aligned_gt[index + 1].coord.x
                                        scaled_x_pos_prev = int(
                                            mask.line.operation.page.page_to_image_scale(x_pos_prev,
                                                                                         PageScaleReference.HIGHRES)) - left
                                        scaled_x_pos_after = int(
                                            mask.line.operation.page.page_to_image_scale(x_pos_after,
                                                                                         PageScaleReference.HIGHRES)) - left
                                        sub_sub_image = sub_image[:, scaled_x_pos_prev: scaled_x_pos_after]

                                        coord = get_best_point(sub_sub_image, scaled_x_pos_prev, left)
                                        print(coord)

                                        x_pos = mask.line.operation.page.image_to_page_scale(coord,
                                                                                             PageScaleReference.HIGHRES)
                                        symbol.coord = t.line.operation.music_line.staff_lines.compute_coord_by_position_in_staff(
                                            x_pos, symbol.position_in_staff)
                                    else:
                                        repeat = True
                        if len(pred) > 0 and pred[0].symbol_type == pred[0].symbol_type.CLEF:
                            pass
                            aligned_gt.insert(0, pred[0])
                        return aligned_gt

                    gt_aligned_symbols = correct_symbols_sequence(gt_aligned_symbols, pred_symbols, t)
                    annotations = Annotations(t.line.operation.page)

                    def match_text_to_symbols(annotations: Annotations, text_line, music_line, neume: List[Neume],
                                              gt_aligned_symbols=None, gt_text=None):
                        page = annotations.page
                        annotations.get_or_create_connection(
                            page.block_of_line(music_line),
                            page.block_of_line(text_line),
                        ).syllable_connections.extend(
                            [SyllableConnector(s.syllable, s.symbols[0]) for s in neume])

                    match_text_to_symbols(annotations, text_line=text_line, music_line=music_line, neume=neumes_of_line)
                    syllables = [s.syllable for s in neumes_of_line]
                    result.append(SingleLinePredictionResult(symbols=gt_aligned_symbols, line=t.line,
                                                             text_line=text_line, text=gt_text,
                                                             annontations=annotations, syllables=syllables))

                    start = True
                if start:
                    index += 1
        if True:
            from shared.pcgtscanvas import PcGtsCanvas

            pages = set(x.line.operation.page.p_id for x in result)
            for i in pages:
                lines = [x for x in result if i == x.line.operation.page.p_id]
                canvas = PcGtsCanvas(lines[0].line.operation.page, PageScaleReference.NORMALIZED_X2)
                for line in lines:
                    # for s in single_line_symbols.symbols:
                    #    s.coord = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(s.coord.x,
                    #                                                                                    s.position_in_staff)
                    canvas.draw(line.symbols, invert=True)
                    canvas.draw(line.annontations)
                    # canvas.draw()
                canvas.show()
        # Todo Not Implemented yet
        yield Result(lines=result)
