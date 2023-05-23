from typing import List

from database import DatabaseBook, DatabasePage
from database.database_book_documents import DatabaseBookDocuments, DocSpanType
from database.file_formats import PcGts
from database.file_formats.book.document import LineMetaInfos
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Connection, SyllableConnector
from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from omr.experimenter.documents.b_evalluator import evaluate_symbols, evaluate_text, \
    evaluate_syllabels, SyllableEvalInput


def prepare_document_gt_symbols(pred_book: DatabaseBook, gt_book: DatabaseBook, ignore=None):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)

    pcgts = []
    docs: List[DocSpanType] = []
    for page in gt_book.pages_with_lock([LockState(Locks.SYMBOLS, True)]):
        pcgts_file = PcGts.from_file(page.file('pcgts'))
        pcgts.append(pcgts_file)
        for i in documents.get_documents_of_page(pcgts_file.page, only_start=True):
            whole_text = i.doc.get_text_of_document(gt_book)
            if "$" in whole_text:
                print("skipping")
                continue
            docs.append(i)

    gt_symbols = []
    gt_meta: List[LineMetaInfos] = []

    pred_symbols = []
    pred_meta: List[LineMetaInfos] = []

    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)
        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())
        if doc:
            gt_pair = i.doc.get_page_line_of_document(gt_book)
            pred_pair = doc.get_page_line_of_document(pred_book)
            print(len(gt_pair))
            print(len(pred_pair))
            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue
            symbols_gt, meta_gt = i.doc.get_symbols(gt_book)
            gt_symbols += symbols_gt
            gt_meta += meta_gt
            symbols_pred, meta_pred = doc.get_symbols(pred_book)
            pred_symbols += symbols_pred
            p_str = ""

            '''
            print("pred")

            for z in symbols_pred:
                l = ""
                for t in z:
                    l += str(t.note_name)
                print(l)
            print("gt")

            for z in symbols_gt:
                l = ""

                for t in z:
                    l += str(t.note_name)
                print(l)
            '''
            pred_meta += meta_pred
    return pred_symbols, gt_symbols


def prepare_document_text(pred_book: DatabaseBook, gt_book: DatabaseBook, debug=False):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)

    pcgts = []
    docs: List[DocSpanType] = []
    for page in gt_book.pages_with_lock([LockState(Locks.SYMBOLS, True)]):
        pcgts_file = PcGts.from_file(page.file('pcgts'))
        pcgts.append(pcgts_file)
        for i in documents.get_documents_of_page(pcgts_file.page, only_start=True):
            whole_text = i.doc.get_text_of_document(gt_book)
            if "$" in whole_text:
                print("skipping")
                continue
            docs.append(i)

    gt_text = []

    pred_text: List[str] = []
    pages = []
    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)

        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())
        if doc:
            gt_text_s = [t.replace("-", "").replace("~", "") for t in i.doc.get_text_list_of_line_document(gt_book)]
            pred_text_s = [t.replace("-", "").replace("~", "") for t in doc.get_text_list_of_line_document(pred_book)]
            if len(pred_text_s) != len(gt_text_s):
                continue
            gt_text += gt_text_s
            pred_text += pred_text_s
            print(i.doc.get_book_u_id())
            pages.append(i.doc.get_book_u_id())
            for p, gt in zip(pred_text, gt_text):
                print(i.doc.get_book_u_id())

                print(f"Pred: {p}")
                print(f"Gt: {gt}")
    for i in pages:
        print(i)
    return pred_text, gt_text


def prepare_document_syllables(pred_book: DatabaseBook, gt_book: DatabaseBook, debug=False):
    eval_data = []
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)
    pcgts = []
    docs: List[DocSpanType] = []
    for page in gt_book.pages_with_lock([LockState(Locks.SYMBOLS, True)]):
        pcgts_file = PcGts.from_file(page.file('pcgts'))
        pcgts.append(pcgts_file)
        for i in documents.get_documents_of_page(pcgts_file.page, only_start=True):
            whole_text = i.doc.get_text_of_document(gt_book)
            if "$" in whole_text:
                print("skipping")
                continue
            docs.append(i)
    for i in docs:
        print(i.doc.get_book_u_id())
        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())
        if doc:
            def get_all_connections_of_music_line(line: Line, connections: List[Connection]):
                syl_connectors: List[SyllableConnector] = []
                for i in connections:
                    for t in i.text_region.lines:
                        if t.id == line.id:  # Todo
                            syl_connectors += i.syllable_connections
                        pass
                return syl_connectors
            gt_pair = i.doc.get_page_line_of_document(gt_book)
            pred_pair = doc.get_page_line_of_document(pred_book)
            print(len(gt_pair))
            print(len(pred_pair))
            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue

            for pred, gt in zip(pred_pair, gt_pair):

                pred_line = pred[0]
                gt_line = gt[0]

                pred_page: DatabasePage = pred[1]
                gt_page: DatabasePage = gt[1]

                gt_annotations = gt_page.pcgts().page.annotations
                pred_annotations = pred_page.pcgts().page.annotations
                gt = get_all_connections_of_music_line(gt_line, gt_annotations.connections)
                pred = get_all_connections_of_music_line(pred_line, pred_annotations.connections)
                if debug:
                    l = ""
                    m = ""
                    for x in gt:
                        l += x.syllable.text
                    for x in pred:
                        m += x.syllable.text
                    print("syllables")
                    print(l)
                    print(m)
                eval_data.append(SyllableEvalInput(pred_annotation=pred_annotations.connections, gt_annotation=gt_annotations.connections, p_line=pred_line, gt_line=gt_line))


    return eval_data


if __name__ == "__main__":
    b = DatabaseBook('mul_2')
    c = DatabaseBook('mul_2_gt')
    # excel_lines1 = evaluate_stafflines(b, c)
    pred_s, gt_s = prepare_document_gt_symbols(b, c)
    excel_lines2 = evaluate_symbols(pred_s, gt_s)
    pred_t, gt_t = prepare_document_text(b, c)
    excel_lines3 = evaluate_text(pred_t, gt_t)
    eval_data = prepare_document_syllables(b, c)
    excel_lines4 = evaluate_syllabels(eval_data)




    from xlwt import Workbook

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    ind = 0
    for x in [excel_lines2, excel_lines3, excel_lines4]:
        ind += 3
        for line in x:
            for ind1, cell in enumerate(line):
                sheet1.write(ind, ind1, str(cell))
            ind += 1
    wb.save("/tmp/eval_data.xls")

