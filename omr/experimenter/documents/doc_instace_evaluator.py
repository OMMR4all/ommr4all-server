from dataclasses import dataclass
from typing import List

from database import DatabaseBook, DatabasePage
from database.database_book_documents import DatabaseBookDocuments, DocSpanType
from database.file_formats import PcGts
from database.file_formats.book.document import LineMetaInfos
from database.file_formats.pcgts import MusicSymbol, Line
from database.file_formats.pcgts.page import Connection, SyllableConnector
from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from omr.experimenter.documents.b_evalluator import evaluate_symbols, evaluate_text, SyllableEvalInput, \
    evaluate_syllabels
from omr.experimenter.documents.evaluater import evaluate_stafflines


def prepare_document(pred_book: DatabaseBook, gt_book: DatabaseBook, ignore=None, ignore_char="$") -> List[DocSpanType]:
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)

    pcgts = []
    docs: List[DocSpanType] = []
    for page in gt_book.pages():
        pcgts_file = PcGts.from_file(page.file('pcgts'))
        pcgts.append(pcgts_file)
        for i in documents.get_documents_of_page(pcgts_file.page, only_start=True):
            whole_text = i.doc.get_text_of_document(gt_book)
            if ignore_char in whole_text:
                print("skipping")
                continue
            docs.append(i)

    return docs


def filter_docs(docs: List[DocSpanType], below_page=253):
    def convert_to_int(st):
        s = ''.join(i for i in st if i.isdigit())
        while len(s) > 0 and s[0] == "0":
            s = s[1:]
        return int(s)

    filtered_docs = []
    for i in docs:
        if convert_to_int(i.doc.start.page_name) < below_page:
            filtered_docs.append(i)
    return filtered_docs

    pass


@dataclass
class LineSymbolEvalData:
    pass
    gt: List[MusicSymbol]
    pred: List[MusicSymbol]


@dataclass
class DocSymbolEvalData:
    eval_symbols: List[LineSymbolEvalData]
    doc_id: str

    def get_doc_symbol_data(self):
        pred_symbols = []
        gt_symbols = []
        for i in self.eval_symbols:
            pred_symbols.append(i.pred)
            gt_symbols.append(i.gt)

        return pred_symbols, gt_symbols


@dataclass
class SymbolEvalData:
    doc_data: List[DocSymbolEvalData]

    def get_symbol_data(self):
        pred_symbols = []
        gt_symbols = []
        for i in self.doc_data:
            pred, gt = i.get_doc_symbol_data()
            if len(pred) == len(gt):
                pred_symbols += pred
                gt_symbols += gt
            else:
                print("Mismatch in len of Lines")
        return pred_symbols, gt_symbols


def gen_eval_symbol_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, docs: List[DocSpanType]):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)
    gt_symbols = []
    gt_meta: List[LineMetaInfos] = []

    pred_symbols = []
    pred_meta: List[LineMetaInfos] = []
    eval_data = []
    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)
        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())

        if doc:
            gt_pair = i.doc.get_page_line_of_document(gt_book)
            pred_pair = doc.get_page_line_of_document(pred_book)
            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue
            symbols_gt, meta_gt = i.doc.get_symbols(gt_book)
            gt_symbols += symbols_gt
            gt_meta += meta_gt
            symbols_pred, meta_pred = doc.get_symbols(pred_book)
            pred_symbols += symbols_pred
            p_str = ""
            eval_lines = []
            for gt, pred in zip(symbols_gt, symbols_pred):
                eval_lines.append(LineSymbolEvalData(pred=pred, gt=gt))

            eval_data.append(DocSymbolEvalData(eval_symbols=eval_lines, doc_id=i.doc.doc_id))
            pred_meta += meta_pred
        else:
            print("Missing")
    return SymbolEvalData(eval_data)


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


def eval_symbols__docs_line_instance(symbol_eval_data: SymbolEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        for f in i.eval_symbols:
            pred, gt = f.pred, f.gt
            excel_data = evaluate_symbols([pred], [gt])
            pred_str = " ".join([t.get_str_representation() for t in pred])
            gt_str = " ".join([t.get_str_representation() for t in gt])
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


@dataclass
class LineTextEvalData:
    gt: str
    pred: str


@dataclass
class DocTextEvalData:
    eval_text: List[LineTextEvalData]
    doc_id: str

    def get_doc_symbol_data(self):
        pred_texts = []
        gt_texts = []
        for i in self.eval_text:
            pred_texts.append(i.pred.replace("j", "i").replace("v", "u"))
            gt_texts.append(i.gt.replace("j", "i").replace("v", "u"))

        return pred_texts, gt_texts

    def get_text_doc_data(self):
        pred_texts = ""
        gt_texts = ""
        for i in self.eval_text:
            pred_texts += i.pred
            gt_texts += i.gt

        return [pred_texts.replace("j", "i").replace("v", "u").replace(" ","")], [gt_texts.replace("j", "i").replace("v", "u").replace(" ","")]


@dataclass
class TextEvalData:
    doc_data: List[DocTextEvalData]

    def get_text_data(self):
        pred_texts = []
        gt_texts = []
        for i in self.doc_data:
            pred, gt = i.get_doc_symbol_data()
            if len(pred) == len(gt):
                pred_texts += pred
                gt_texts += gt
            else:
                print("Mismatch in len of Lines")
        return pred_texts, gt_texts

    def get_text_data_ignore_line_endings(self):
        pred_texts = []
        gt_texts = []
        for i in self.doc_data:
            pred, gt = i.get_text_doc_data()
            if len(pred) == len(gt):
                pred_texts += pred
                gt_texts += gt
            else:
                print("Mismatch in len of Lines")
        return pred_texts, gt_texts


def gen_eval_text_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, docs: List[DocSpanType]):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)

    eval_data = []
    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)
        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())

        if doc:
            gt_pair = i.doc.get_page_line_of_document(gt_book)
            pred_pair = doc.get_page_line_of_document(pred_book)
            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue
            text_gt = i.doc.get_text_list_of_line_document(gt_book)

            text_pred = doc.get_text_list_of_line_document(pred_book)
            eval_lines = []
            for gt, pred in zip(text_gt, text_pred):
                print("pred")
                print(pred)
                print(gt)
                eval_lines.append(LineTextEvalData(pred=pred.replace("-", "").lower(), gt=gt.replace("-", "").lower()))

            eval_data.append(DocTextEvalData(eval_text=eval_lines, doc_id=i.doc.doc_id))
        else:
            print("Missing")
    return TextEvalData(eval_data)


def eval_texts_line_instance(symbol_eval_data: TextEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        for f in i.eval_text:
            pred, gt = f.pred, f.gt
            excel_data = evaluate_text([pred], [gt])
            pred_str = pred
            gt_str = gt
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


def eval_texts_docs_instance(symbol_eval_data: TextEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        pred, gt = i.get_doc_symbol_data()
        excel_data = evaluate_text(pred, gt)
        docs_instance_eval_data.append([(pred[0], gt[0]), excel_data, i.doc_id])
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


def eval_texts_docs_instance_ignore_lines(symbol_eval_data: TextEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        pred, gt = i.get_text_doc_data()
        excel_data = evaluate_text(pred, gt)
        pred_str = " ".join([iz for iz in pred])
        gt_str = " ".join([iz for iz in gt])
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


@dataclass
class LineSyllableEvalData:
    syllable_eval_data: SyllableEvalInput
    pred_txt: str
    gt_txt: str


@dataclass
class DocSyllableEvalData:
    eval_syls: List[LineSyllableEvalData]
    doc_id: str

    def get_doc_syl_data(self):
        eval_data = []
        for i in self.eval_syls:
            eval_data.append(i.syllable_eval_data)

        return eval_data

    def get_pred_str(self):
        pred = ""
        for i in self.eval_syls:
            pred += i.pred_txt
        return pred
        pass

    def get_gt_str(self):
        gt = ""
        for i in self.eval_syls:
            gt += i.gt_txt
        return gt


@dataclass
class SyllableEvalData:
    doc_data: List[DocSyllableEvalData]

    def get_syl_data(self):
        eval_data = []
        for i in self.doc_data:
            eval_data += i.get_doc_syl_data()
        return eval_data


def gen_eval_syllable_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, docs: List[DocSpanType]):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)
    gt_text = []
    debug = False
    pred_text = []
    eval_data = []
    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)
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

            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue
            eval_lines = []

            for pred, gt in zip(pred_pair, gt_pair):

                pred_line: Line = pred[0]
                gt_line: Line = gt[0]

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
                eval_lines.append(LineSyllableEvalData(SyllableEvalInput(pred_annotation=pred_annotations.connections,
                                                                         gt_annotation=gt_annotations.connections,
                                                                         p_line=pred_line, gt_line=gt_line),
                                                       pred_txt=pred_line.text(), gt_txt=gt_line.text())
                                  ,
                                  )
            eval_data.append(DocSyllableEvalData(eval_syls=eval_lines, doc_id=i.doc.doc_id))
    return SyllableEvalData(eval_data)


def eval_syl_docs_instance(symbol_eval_data: SyllableEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        data = i.get_doc_syl_data()
        excel_data = evaluate_syllabels(data)
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


def eval_syl_docs_line_instance(symbol_eval_data: SyllableEvalData, sheet):
    docs_instance_eval_data = []
    for i in symbol_eval_data.doc_data:
        for t in i.eval_syls:
            data = [t.syllable_eval_data]
            excel_data = evaluate_syllabels(data)
            pred_str = t.pred_txt
            gt_str = t.gt_txt
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


@dataclass
class LineLayoutEvalData:
    gt: Line
    pred: Line
    pred_txt: str
    gt_txt: str


@dataclass
class DocLayoutEvalData:
    eval_layout: List[LineLayoutEvalData]
    doc_id: str

    def get_doc_layout_data(self):

        return self.eval_layout

    def get_pred_str(self):
        pred = ""
        for i in self.eval_layout:
            pred += i.pred_txt
        return pred
        pass

    def get_gt_str(self):
        gt = ""
        for i in self.eval_layout:
            gt += i.gt_txt
        return gt


@dataclass
class LayoutEvalData:
    doc_data: List[DocLayoutEvalData]

    def get_layout_eval_data(self):
        eval_data = []
        for i in self.doc_data:
            eval_data += i.get_doc_layout_data()
        return eval_data


def eval_layout(eval_data: List[LineLayoutEvalData]):
    tp = 0
    fp = 0
    fn = 0

    for i in eval_data:
        tp_l = 0
        for s in i.gt.symbols:
            if i.pred.coords.polygon_contains_point(s.coord):
                tp_l += 1
                continue

        fp += (len(i.gt.symbols) - tp_l)
        fn += (len(i.gt.symbols) - tp_l)
        tp += tp_l
    p = tp / (tp + fp) if (tp + fp) > 0 else 1
    r = tp / (tp + fn) if (tp + fn) > 0 else 1
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 1
    acc = tp / (tp + fp + fn) if (tp + fn + fp) > 0 else 1
    excel_lines = []
    labels = ["Tp", "Fn", "FP", "F1", "acc", "rec", "prec"]
    results = [tp, fn, fp, f1, acc, r, p]
    excel_lines.append(labels)
    excel_lines.append(results)
    return excel_lines


def gen_eval_layout_documents_data(pred_book: DatabaseBook, gt_book: DatabaseBook, docs: List[DocSpanType]):
    documents = DatabaseBookDocuments().load(gt_book)
    documents_pred = DatabaseBookDocuments().load(pred_book)
    gt_text = []
    debug = False
    pred_text = []
    eval_data = []
    for i in docs:
        # doc = documents_pred.database_documents.get_document_by_id(i.doc.doc_id)
        doc = documents_pred.database_documents.get_document_by_b_uid(i.doc.get_book_u_id())

        if doc:

            gt_pair = i.doc.get_page_line_of_document(gt_book)
            pred_pair = doc.get_page_line_of_document(pred_book)

            if len(gt_pair) != len(pred_pair):
                print("skipping syll")
                continue
            eval_lines = []

            for pred, gt in zip(pred_pair, gt_pair):
                pred_line: Line = pred[0]
                gt_line: Line = gt[0]

                pred_page: DatabasePage = pred[1]
                gt_page: DatabasePage = gt[1]
                ml_gt = gt_page.pcgts().page.closest_music_line_to_text_line(gt_line)
                ml_pred = pred_page.pcgts().page.closest_music_line_to_text_line(pred_line)

                eval_lines.append(
                    LineLayoutEvalData(pred_txt=pred_line.text(), gt_txt=gt_line.text(), gt=ml_gt, pred=ml_pred)
                    ,
                )
            eval_data.append(DocLayoutEvalData(eval_layout=eval_lines, doc_id=i.doc.doc_id))
    return LayoutEvalData(eval_data)


def eval_layout_docs_line_instance(layout_eval_data: LayoutEvalData, sheet):
    docs_instance_eval_data = []
    for i in layout_eval_data.doc_data:
        for t in i.eval_layout:
            data = t
            excel_data = eval_layout([data])
            pred_str = t.pred_txt
            gt_str = t.gt_txt
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


def eval_layout_docs_instance(layout_eval_data: LayoutEvalData, sheet):
    docs_instance_eval_data = []
    for i in layout_eval_data.doc_data:
        data = i.get_doc_layout_data()
        excel_data = eval_layout(data)
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


def write_staffline_eval_data(excel_line, sheet):
    ind = 3
    left = 3
    for ind_d, d in enumerate(excel_line):

        for ind1, cell in enumerate(d):
            sheet.write(ind, ind1 + left, cell)
        ind += 1


if __name__ == "__main__":

    def overview_excel_sheet(excel_lines1, layout_eval_data, symbol_eval_data, text_eval_data, syl_eval_data, sheet):

        ex_layout = eval_layout(layout_eval_data.get_layout_eval_data())
        pred, gt = symbol_eval_data.get_symbol_data()
        excel_symbol = evaluate_symbols(pred, gt)

        pred, gt = text_eval_data.get_text_data()
        excel_text = evaluate_text(pred, gt)
        pred, gt = text_eval_data.get_text_data_ignore_line_endings()
        excel_text2 = evaluate_text(pred, gt)
        ex_syll = evaluate_syllabels(syl_eval_data.get_syl_data())
        ind = 3
        left = 3
        for t in [excel_lines1, ex_layout, excel_symbol, excel_text, excel_text2, ex_syll]:

            for ind_d, d in enumerate(t):

                for ind1, cell in enumerate(d):
                    sheet.write(ind, ind1 + left, cell)
                ind += 1
        pass


    b = DatabaseBook('mul_2_c')
    c = DatabaseBook('mul_2_rsync_gt')
    # excel_lines1 = evaluate_stafflines(b, c)
    docs = prepare_document(b, c)
    docs = filter_docs(docs, 253)
    layout_eval_data = gen_eval_layout_documents_data(b, c, docs)

    symbol_eval_data = gen_eval_symbol_documents_data(b, c, docs)
    text_eval_data = gen_eval_text_documents_data(b, c, docs)
    syl_eval_data = gen_eval_syllable_documents_data(b, c, docs)

    from xlwt import Workbook

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    excel_lines1 = evaluate_stafflines(b, c)
    sheet0 = wb.add_sheet('Stafflines')
    write_staffline_eval_data(excel_lines1, sheet0)
    sheet03 = wb.add_sheet('Layout Docs')
    eval_layout_docs_instance(layout_eval_data, sheet03)
    sheet04 = wb.add_sheet('Layout Lines')
    eval_layout_docs_line_instance(layout_eval_data, sheet04)

    sheet1 = wb.add_sheet('Symbols Docs')
    eval_symbols__docs_instance(symbol_eval_data, sheet1)
    sheet2 = wb.add_sheet('Symbols Lines')
    eval_symbols__docs_instance(symbol_eval_data, sheet2)
    sheet3 = wb.add_sheet('Text Docs')
    eval_texts_docs_instance(text_eval_data, sheet3)
    sheet4 = wb.add_sheet('Text Lines')
    eval_texts_line_instance(text_eval_data, sheet4)
    sheet34 = wb.add_sheet('Text Docs Line Ignored')
    eval_texts_docs_instance_ignore_lines(text_eval_data, sheet34)
    sheet5 = wb.add_sheet('Syllable Docs')
    eval_syl_docs_instance(syl_eval_data, sheet5)
    sheet6 = wb.add_sheet('Syllable Line')
    eval_syl_docs_line_instance(syl_eval_data, sheet6)

    sheet10 = wb.add_sheet('overview 1')
    overview_excel_sheet(excel_lines1, layout_eval_data, symbol_eval_data, text_eval_data, syl_eval_data, sheet10)
    wb.save("/tmp/eval_data.xls")
