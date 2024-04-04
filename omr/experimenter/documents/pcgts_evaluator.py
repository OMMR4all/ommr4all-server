import os
from difflib import SequenceMatcher
from typing import List

import edlib
from PIL import ImageDraw, Image

from database import DatabaseBook, DatabasePage
from database.database_book import logger
from database.file_formats import PcGts
from database.file_formats.book.document import LineMetaInfos
from database.file_formats.pcgts import Line, SymbolType, MusicSymbol, Page, PageScaleReference
from database.file_formats.pcgts.page import SyllableConnector, Connection, Annotations
from omr.experimenter.documents.b_evalluator import SyllableEvalInput, evaluate_syllabels, evaluate_symbols
from omr.experimenter.documents.doc_instace_evaluator import eval_syl_docs_instance, \
    eval_syl_docs_line_instance, prepare_document, LineSymbolEvalData, DocSymbolEvalData, SymbolEvalData, \
    LineSyllableEvalData, DocSyllableEvalData, SyllableEvalData
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step
from database.model import Model, MetaId

from omr.steps.symboldetection.evaluator import Codec

def ocr_based_filterfunction(db_page: DatabasePage, db_page_pred: DatabasePage):

    meta = Step.meta(AlgorithmTypes.OCR_GUPPY)
    # model = meta.best_model_for_book(book)
    from ommr4all.settings import BASE_DIR
    from omr.steps.text.predictor import PredictionResult as TextPredictionResult

    model = Model(
        MetaId.from_custom_path(BASE_DIR + '/storage/Graduel_Syn/models/text_guppy/2023-10-05T16:39:40/', meta.type()))
    # model = Model.from_id_str()
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    ocr_predictor = meta.create_predictor(settings)
    conf = 99999
    same = False
    same_both = False
    for i in ocr_predictor.predict([db_page]):
        ocr_r: TextPredictionResult = i
        best_pred = None

        for t in ocr_r.text_lines:
            pred = [(t, pos) for t, pos in t.text if t not in ' -']
            syls = t.line.operation.text_line.sentence.syllables
            #print(pred)
            #print([i.text for i in syls])
            #print("".join([i.text for i in syls]))
            #print("".join([i[0] for i in pred]))
            ed = edlib.align("".join([i[0] for i in pred]), "".join([i.text for i in syls]), mode="NW", task="path")
            if ed["editDistance"] < conf and len("".join([i.text for i in syls])) > 0:
                gt_string = "".join([i.text for i in syls])
                print(f"conf:  {ed['editDistance']} : length = {len(gt_string)}")
                conf = ed["editDistance"]
                best_pred = t.text
                if conf == 0:# / len(gt_string) < 0.05:
                    print(f"confT:  {ed['editDistance']} : length = {len(gt_string)}")

                    same = True
    if same:
        conf = 99999

        for i in ocr_predictor.predict([db_page_pred]):
            ocr_r: TextPredictionResult = i
            best_pred = None

            for t in ocr_r.text_lines:
                pred = [(t, pos) for t, pos in t.text if t not in ' -']
                syls = t.line.operation.text_line.sentence.syllables
                # print(pred)
                # print([i.text for i in syls])
                # print("".join([i.text for i in syls]))
                # print("".join([i[0] for i in pred]))
                ed = edlib.align("".join([i[0] for i in pred]), "".join([i.text for i in syls]), mode="NW", task="path")
                if ed["editDistance"] < conf and len("".join([i.text for i in syls])) > 0:
                    conf = ed["editDistance"]
                    best_pred = t.text
                    gt_string = "".join([i.text for i in syls])
                    print(f"conf2:  {ed['editDistance']} : length = {len(gt_string)}")

                    if conf == 0: #conf / len(gt_string) < 0.05:

                        same_both = True
    return same_both
def filterfunction(db_page: DatabasePage, pred_page: DatabasePage):
    if db_page.page_progress().verified_allowed():
        for i in db_page.pcgts().page.all_music_lines():
            for t in i.symbols:
                if t.missing:
                    return False
    else:
        return False

    if ocr_based_filterfunction(db_page, pred_page):
        return True

    return False


def gen_eval_symbol_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, filterf=None):
    pages_pred = pred_book.pages()
    pages_gt = gt_book.pages()
    pages_gt_names = [x.page for x in pages_gt]
    pages_pred = [x for x in pages_pred if x.page in pages_gt_names]
    eval_data = []
    for p, gt in zip(pages_pred, pages_gt):
        if p.page != gt.page:
            print(f"skipping {p.page} {gt.page}")
            continue

        if filterf and not filterf(gt, p):
            continue
        p_lines = p.pcgts().page.all_music_lines()
        gt_lines = gt.pcgts().page.all_music_lines()
        eval_lines = []

        for p_line, gt_line in zip(p_lines, gt_lines):
            eval_lines.append(LineSymbolEvalData(pred=p_line.symbols, gt=gt_line.symbols))
        eval_data.append(DocSymbolEvalData(eval_symbols=eval_lines, doc_id=p.pcgts().page.location.page, reference=[p, gt]))

    return SymbolEvalData(eval_data)


def gen_eval_syllable_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, debug=False, filterf=None):
    pages_pred = pred_book.pages()
    pages_gt = gt_book.pages()
    pages_gt_names = [x.page for x in pages_gt]
    pages_pred = [x for x in pages_pred if x.page in pages_gt_names]

    def get_all_connections_of_music_line(line: Line, connections: List[Connection]):
        syl_connectors: List[SyllableConnector] = []
        for i in connections:
            for t in i.text_region.lines:
                if t.id == line.id:  # Todo
                    syl_connectors += i.syllable_connections
                pass
        return syl_connectors

    eval_data = []
    for p, gt in zip(pages_pred, pages_gt):
        p: DatabasePage = p
        if p.page != gt.page:
            print(f"skipping {p.page} {gt.page}")
            continue
        if filterf and not filterf(gt, p):
            continue
        p_lines = p.pcgts().page.all_text_lines()
        gt_lines = gt.pcgts().page.all_text_lines()
        eval_lines = []

        for p_line, gt_line in zip(p_lines, gt_lines):
            pred_line: Line = p_line
            gt_line: Line = gt_line

            pred_page: DatabasePage = p
            gt_page: DatabasePage = gt

            gt_annotations = gt_page.pcgts().page.annotations
            pred_annotations = pred_page.pcgts().page.annotations
            gtd = get_all_connections_of_music_line(gt_line, gt_annotations.connections)
            predd = get_all_connections_of_music_line(pred_line, pred_annotations.connections)
            if debug:
                l = ""
                m = ""
                for x in gtd:
                    l += x.syllable.text
                for x in predd:
                    m += x.syllable.text
                print("syllables")
                print(l)
                print(m)
            eval_lines.append(LineSyllableEvalData(SyllableEvalInput(pred_annotation=pred_annotations.connections,
                                                                     gt_annotation=gt_annotations.connections,
                                                                     p_line=pred_line, gt_line=gt_line,
                                                                     gt_page=gt_page.page, pred_page=pred_page.page),
                                                   pred_txt=pred_line.text(), gt_txt=gt_line.text(),
                                                   gt_page=gt_page.page, pred_page=pred_page.page))
        eval_data.append(DocSyllableEvalData(eval_syls=eval_lines, doc_id=p.pcgts().page.location.page, reference = [p, gt]))

    return SyllableEvalData(eval_data)


if __name__ == "__main__":

    # def overview_excel_sheet(excel_lines1, layout_eval_data, symbol_eval_data, text_eval_data, syl_eval_data, sheet):
    #
    #     ex_layout = eval_layout(layout_eval_data.get_layout_eval_data())
    #     pred, gt = symbol_eval_data.get_symbol_data()
    #     excel_symbol = evaluate_symbols(pred, gt)
    #
    #     pred, gt = text_eval_data.get_text_data()
    #     excel_text = evaluate_text(pred, gt)
    #     pred, gt = text_eval_data.get_text_data_ignore_line_endings()
    #     excel_text2 = evaluate_text(pred, gt)
    #     ex_syll, errors = evaluate_syllabels(syl_eval_data.get_syl_data())
    #     ind = 3
    #     left = 3
    #     for t in [excel_lines1, ex_layout, excel_symbol, excel_text, excel_text2, ex_syll]:
    #
    #         for ind_d, d in enumerate(t):
    #
    #             for ind1, cell in enumerate(d):
    #                 sheet.write(ind, ind1 + left, cell)
    #             ind += 1
    #     pass
    def visualize_symbol_errors(symbol_eval_data: SymbolEvalData):
        def pass_as(pred, gt, codec, symbolsgt, symbolspred):
            tp_symbols = []
            symbols_missing = []
            symbols_additional = []
            symbols_position_fp = []
            symbols_position_fn = []
            symbols_note_con_fp = []
            symbols_note_con_fn = []

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
                    tp_symbols += symbolsgt[gt_start:gt_end]
                elif opcode == 'insert' or opcode == 'replace' or opcode == 'delete':
                    total_errors += pred_end - pred_start + gt_end - gt_start
                    for i, s in enumerate(gt[gt_start:gt_end]):
                        entry = codec.codec[s]
                        symbol_type = entry[0]
                        if symbol_type == SymbolType.ACCID:
                            missing_accids += 1
                            symbols_missing.append(symbolsgt[gt_start + i])
                        elif symbol_type == SymbolType.NOTE:
                            if opcode == 'replace' and pred_end > pred_start + i:
                                # check for wrong connection
                                p = codec.codec[pred[pred_start + i]]
                                if p[0] == symbol_type:
                                    if p[3] == entry[3]:
                                        wrong_position_in_staff += 1
                                        symbols_position_fp.append(symbolsgt[gt_start + i])
                                    else:
                                        wrong_note_connections += 1
                                        symbols_note_con_fp.append(symbolsgt[gt_start + i])
                                else:
                                    missing_notes += 1
                                    symbols_missing.append(symbolsgt[gt_start + i])
                            else:
                                missing_notes += 1
                                symbols_missing.append(symbolsgt[gt_start + i])
                        elif symbol_type == SymbolType.CLEF:
                            missing_clefs += 1
                            symbols_missing.append(symbolsgt[gt_start + i])
                        else:
                            raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

                    for i, s in enumerate(pred[pred_start:pred_end]):
                        entry = codec.codec[s]
                        symbol_type = entry[0]
                        if symbol_type == SymbolType.ACCID:
                            additional_accid += 1
                            symbols_additional.append(symbolspred[pred_start + i])
                        elif symbol_type == SymbolType.NOTE:
                            if opcode == 'replace' and gt_end > gt_start + i:
                                # check for wrong connection
                                p = codec.codec[gt[gt_start + i]]
                                if p[0] == symbol_type:
                                    if p[3] == entry[3]:
                                        add_wrong_pos_in_staff += 1
                                        symbols_position_fn.append(symbolspred[pred_start + i])
                                    else:
                                        add_wrong_note_con += 1
                                        symbols_note_con_fn.append(symbolspred[pred_start + i])
                                else:
                                    additional_note += 1
                                    symbols_additional.append(symbolspred[pred_start + i])
                            else:
                                additional_note += 1
                                symbols_additional.append(symbolspred[pred_start + i])
                        elif symbol_type == SymbolType.CLEF:
                            additional_clef += 1
                            symbols_additional.append(symbolspred[pred_start + i])
                        else:
                            raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

                else:
                    raise ValueError(opcode)
            return tp_symbols, symbols_missing, symbols_additional, symbols_position_fp, symbols_position_fn, symbols_note_con_fp, symbols_note_con_fn

        #codec = Codec()
        def draw_symbols(image: ImageDraw, symbols: List[MusicSymbol], color=(0, 0, 255), page: Page= None):
            for s in symbols:
                s: MusicSymbol
                coord = page.page_to_image_scale(s.coord, PageScaleReference.NORMALIZED_X2)

                image.rectangle((  coord.x - 5, coord.y - 5,coord.x+5, coord.y+5,), outline=color, width=2)


        for i in symbol_eval_data.doc_data:
            pred_s, gt_s = i.get_doc_symbol_data()
            pred_page, gt_page = i.reference
            pred_page: DatabasePage = pred_page
            gt_page: DatabasePage = gt_page
            image = Image.open(gt_page.pcgts().page.location.file("color_norm_x2").local_path())
            draw = ImageDraw.Draw(image)
            for gt, pred in zip(gt_s, pred_s):
                codec = Codec()

                #gt_sequence = codec.symbols_to_label_sequence(gt, False)
                #pred_sequence = codec.symbols_to_label_sequence(pred, False)
                gt_sequence_nc = codec.symbols_to_label_sequence(gt, True)
                pred_sequence_nc = codec.symbols_to_label_sequence(pred, True)
                tp_symbols, symbols_missing, symbols_additional, symbols_position_fp, symbols_position_fn, symbols_note_con_fp, symbols_note_con_fn = pass_as(pred_sequence_nc, gt_sequence_nc, codec, gt, pred)
                draw_symbols(draw, symbols_missing, (255, 0, 0), gt_page.pcgts().page)
                draw_symbols(draw, symbols_additional, (0, 0, 255), gt_page.pcgts().page)
                draw_symbols(draw, symbols_position_fp, (0, 255, 0), gt_page.pcgts().page)
                draw_symbols(draw, symbols_position_fn, (0, 255, 255), gt_page.pcgts().page)
                draw_symbols(draw, symbols_note_con_fp, (255, 255, 0), gt_page.pcgts().page)
                draw_symbols(draw, symbols_note_con_fn, (255, 255, 255), gt_page.pcgts().page)
            if not os.path.exists("/tmp/symbols"):
                os.mkdir("/tmp/symbols")

            image.save(f"/tmp/symbols/{gt_page.page}.png")

    def eval_symbols__docs_instance(symbol_eval_data: SymbolEvalData, sheet):
        docs_instance_eval_data = []
        for i in symbol_eval_data.doc_data:
            pred, gt = i.get_doc_symbol_data()
            excel_data = evaluate_symbols(pred, gt)
            pred_str = " ".join([t.get_str_representation() for iz in pred for t in iz])
            gt_str = " ".join([t.get_str_representation() for iz in gt for t in iz])
            docs_instance_eval_data.append([(pred_str, gt_str), excel_data])
        ind = 3
        left = 3
        first = False
        for ind_d, d in enumerate(docs_instance_eval_data):
            eval_data1, lines = d
            p_str, gt_str = eval_data1
            if not first:
                for ind2, line in enumerate(lines[:-1]):
                    sheet.write(ind, 0, "Doc_id")
                    sheet.write(ind, 1, "p_str")
                    sheet.write(ind, 2, "gt_str")
                    for ind1, cell in enumerate(line):
                        sheet.write(ind, ind1 + left, str(cell))
                    ind += 1
                first = True
            sheet.write(ind, 0, str(symbol_eval_data.doc_data[ind_d].doc_id))
            sheet.write(ind, 1, str(p_str))
            sheet.write(ind, 2, str(gt_str))

            for ind1, cell in enumerate(lines[-1]):
                sheet.write(ind, ind1 + left, cell)
            ind += 1


    def eval_syl_docs_instance(symbol_eval_data: SyllableEvalData, sheet):
        docs_instance_eval_data = []
        for i in symbol_eval_data.doc_data:
            data = i.get_doc_syl_data()
            excel_data, errors = evaluate_syllabels(data)
            pred_str = i.get_pred_str()
            gt_str = i.get_gt_str()
            docs_instance_eval_data.append([(pred_str, gt_str), excel_data, i.doc_id])
        ind = 3
        left = 3
        first = False
        for ind_d, d in enumerate(docs_instance_eval_data):
            eval_data1, lines, docs_id = d
            p_str, gt_str = eval_data1
            if not first:
                for ind2, line in enumerate(lines[:-1]):
                    sheet.write(ind, 0, "Doc_id")
                    sheet.write(ind, 1, "p_str")
                    sheet.write(ind, 2, "gt_str")
                    for ind1, cell in enumerate(line):
                        sheet.write(ind, ind1 + left, str(cell))
                    ind += 1
                first = True
            sheet.write(ind, 0, str(docs_id))
            sheet.write(ind, 1, str(p_str))
            sheet.write(ind, 2, str(gt_str))

            for ind1, cell in enumerate(lines[-1]):
                sheet.write(ind, ind1 + left, cell)
            ind += 1


    def visualize_syllable_errors(syl_eval_data: SyllableEvalData):
        def get_all_connections_of_music_line(line: Line, connections: List[Connection]) -> List[SyllableConnector]:
            syl_connectors: List[SyllableConnector] = []
            for i in connections:
                for t in i.text_region.lines:
                    if t.id == line.id:  # Todo
                        syl_connectors += i.syllable_connections
                    pass
            return syl_connectors

        def pass_as(line):
            tp_l = []
            fp_l = []
            fn_l = []
            tp = 0
            fp = 0
            fn = 0

            fp_prev_line = []
            for i in line:
                pred_annotations = i.pred_annotation
                gt_annotations = i.gt_annotation
                p_line = i.p_line
                gt_line = i.gt_line

                syl_connectors_pred = get_all_connections_of_music_line(p_line, pred_annotations)
                syl_connectors_gt = get_all_connections_of_music_line(gt_line, gt_annotations)
                add_gt_connections = []

                for i in syl_connectors_gt:
                    found = False
                    index = -1

                    for ind, p in enumerate(syl_connectors_pred):
                        if abs(i.note.coord.x - p.note.coord.x) < 0.005:
                            # (i.syllable.text in p.syllable.text or p.syllable.text in i.syllable.text) when different grammar used to split words in syllabels
                            if i.syllable.text.lower() == p.syllable.text.lower() or (
                                    i.syllable.text.lower() in p.syllable.text.lower() or p.syllable.text.lower() in i.syllable.text.lower()):
                                tp += 1
                                tp_l.append(i)
                                found = True
                                del syl_connectors_pred[ind]
                                break
                            else:
                                pass
                                # print(f'gt {i.syllable.text} p: {p.syllable.text}')
                    if not found:
                        add_gt_connections.append(i)
                        fn += 1
                        fn_l.append(i)

                fp += len(syl_connectors_pred)
                for it in syl_connectors_pred:
                    fp_l.append(it)
            return tp_l, fp_l, fn_l
        def draw_syls(image: ImageDraw, conns: List[SyllableConnector], color=(0, 0, 255), page: Page= None,offset=5):
            for s in conns:
                coord = page.page_to_image_scale(s.note.coord, PageScaleReference.NORMALIZED_X2)
                image.rectangle((  coord.x - 5, coord.y - 5,coord.x+5, coord.y+5,), outline=color, width=2)
                image.text((  coord.x, coord.y + offset), text=s.syllable.text, fill=color, stroke_width=2)
        for i in syl_eval_data.doc_data:
            x = i.get_doc_syl_data()
            pred_page, gt_page = i.reference
            pred_page: DatabasePage = pred_page
            gt_page: DatabasePage = gt_page
            image = Image.open(gt_page.pcgts().page.location.file("color_norm_x2").local_path())
            draw = ImageDraw.Draw(image)
            #for t in x:

            tp_symbols, fp_syl, fn_syl  = pass_as(x)

            draw_syls(draw, fp_syl, (0, 255, 0), gt_page.pcgts().page)
            draw_syls(draw, fn_syl, (0, 0, 255), gt_page.pcgts().page, offset=10)
            if not os.path.exists("/tmp/syls"):

                os.mkdir("/tmp/syls")
            image.save(f"/tmp/syls/{gt_page.page}.png")

    def overview_excel_sheet(symbol_eval_data, syl_eval_data, sheet):

        pred, gt = symbol_eval_data.get_symbol_data()
        excel_symbol = evaluate_symbols(pred, gt)
        ex_syll, errors = evaluate_syllabels(syl_eval_data.get_syl_data())
        ind = 3
        left = 3
        for t in [excel_symbol, ex_syll]:

            for ind_d, d in enumerate(t):

                for ind1, cell in enumerate(d):
                    sheet.write(ind, ind1 + left, cell)
                ind += 1
        pass


    logger.info("Successfully imported Lyrics database into memory")
    # books = ['mul_2_23_1_24_Evaluation']
    books = ['Graduel_Syn']
    for i in books:
        b = DatabaseBook(i)
        c = DatabaseBook('Graduel_Syn22_03_24')
        # excel_lines1 = evaluate_stafflines(b, c)

        symbol_eval_data = gen_eval_symbol_documents_data(b, c, filterf=filterfunction)
        syl_eval_data = gen_eval_syllable_documents_data(b, c, filterf=filterfunction)

        from xlwt import Workbook

        wb2 = Workbook()
        # Workbook is created
        wb = Workbook()
        visualize_symbol_errors(symbol_eval_data)
        visualize_syllable_errors(syl_eval_data)

        #exit()
        sheet1 = wb.add_sheet('Symbols Docs')
        eval_symbols__docs_instance(symbol_eval_data, sheet1)
        sheet2 = wb.add_sheet('Symbols Lines')
        eval_symbols__docs_instance(symbol_eval_data, sheet2)

        sheet5 = wb.add_sheet('Syllable Docs')
        eval_syl_docs_instance(syl_eval_data, sheet5)
        sheet6 = wb.add_sheet('Syllable Line')
        eval_syl_docs_line_instance(syl_eval_data, sheet6)
        sheet10 = wb.add_sheet('overview 1')
        overview_excel_sheet(symbol_eval_data, syl_eval_data, sheet10)

        # print(f"Skipped {skipped}, total, {len(docs)}")
        wb.save(f"/tmp/{i}.xls")
