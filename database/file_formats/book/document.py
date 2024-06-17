import enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import uuid

import ezodf
import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin

from database import DatabasePage, DatabaseBook
from PIL import Image

from database.file_formats.pcgts import PageScaleReference, MusicSymbol, Line
from database.file_formats.pcgts.page import Sentence


class DatasetSource(str, enum.Enum):
    GR = "gregorianik_gr"
    AN = "gregorianik_an"
    CM = "corpus_monodicum"
    CT = "cantus_db"


@dataclass
class DocumentMetaInfos(DataClassJSONMixin):
    cantus_id: str = ""
    initium: str = ""
    genre: str = ""
    url: str = ""
    dataset_source: DatasetSource = None
    festum: str = ""
    dies: str = ""
    extended_source: str = ""
    manuscript: str = ""

    pass


@dataclass
class LineMetaInfos:
    line_id: str
    left_p: float
    right_p: float
    page: str


class DocumentConnection:
    def __init__(self, page_id=None, page_name=None, line_id=None, row: int = None):
        self.page_id = page_id
        self.page_name = page_name
        self.line_id = line_id
        self.row = row

    @staticmethod
    def from_json(json: dict):
        return DocumentConnection(
            json.get('page_id', None),
            json.get('page_name', None),
            json.get('line_id', None),
            json.get('row', None),
        )

    def to_json(self):
        return {
            "page_id": self.page_id,
            "page_name": self.page_name,
            "line_id": self.line_id,
            "row": self.row,
        }

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Document:
    def __init__(self, page_ids, page_names, start: DocumentConnection, end: DocumentConnection,
                 monody_id=None, doc_id=None, textinitium='', textline_count=0,
                 document_meta_infos: DocumentMetaInfos = None):
        self.monody_id = monody_id if monody_id else str(uuid.uuid4())
        self.doc_id = doc_id if doc_id else str(uuid.uuid4())
        self.pages_ids: List[int] = page_ids
        self.pages_names: List[str] = page_names
        self.start: DocumentConnection = start
        self.end: DocumentConnection = end
        self.textinitium = textinitium
        self.textline_count = textline_count
        self.document_meta_infos = document_meta_infos

    @staticmethod
    def from_json(json: dict):
        return Document(
            page_ids=json.get('page_ids', []),
            page_names=json.get('pages_names', []),
            monody_id=json.get('monody_id', None),
            doc_id=json.get('doc_id', None),
            start=DocumentConnection.from_json(json.get('start_point', None)),
            end=DocumentConnection.from_json(json.get('end_point', None)),
            textinitium=json.get('textinitium', ''),
            textline_count=json.get('textline_count', ''),
            document_meta_infos=DocumentMetaInfos.from_dict(json.get('document_meta_infos', None)) if json.get(
                'document_meta_infos', None) is not None else DocumentMetaInfos(),

        )

    def to_json(self):
        return {
            "page_ids": self.pages_ids,
            "pages_names": self.pages_names,
            "monody_id": self.monody_id,
            "doc_id": self.doc_id,
            "start_point": self.start.to_json(),
            "end_point": self.end.to_json(),
            "textinitium": self.textinitium,
            "textline_count": self.textline_count,
            "document_meta_infos": self.document_meta_infos.to_dict() if self.document_meta_infos else DocumentMetaInfos().to_dict()
        }

    def get_database_pages(self, book):
        return [DatabasePage(book, x) for x in self.pages_names]
    def export_to_ods(self, filename, editor):
        from database.file_formats.exporter.monodi.ods import MonodiOdsConfig
        from ezodf import newdoc, Paragraph, Heading, Sheet
        ods = newdoc(doctype='ods', filename=filename)
        config = MonodiOdsConfig()
        sheet = ezodf.Sheet('Tabellenblatt1', size=(2, config.length))
        ods.sheets += sheet

        for x in config.entries:
            sheet[x.cell.get_entry()].set_value(x.value)
        sheet[''.join([config.dict['Textinitium Editionseinheit'].cell.column, str(2)])].set_value(self.textinitium)
        sheet[''.join([config.dict['Startseite'].cell.column, str(2)])].set_value(self.start.page_name)
        sheet[''.join([config.dict['Startzeile'].cell.column, str(2)])].set_value(self.start.row)
        sheet[''.join([config.dict['Endseite'].cell.column, str(2)])].set_value(self.end.page_name)
        sheet[''.join([config.dict['Endzeile'].cell.column, str(2)])].set_value(self.end.row)
        sheet[''.join([config.dict['Editor'].cell.column, str(2)])].set_value(str(editor))
        sheet[''.join([config.dict['Doc-Id\' (intern)'].cell.column, str(2)])].set_value(self.monody_id)
        sheet[''.join([config.dict['Quellen-ID (intern)'].cell.column, str(2)])].set_value('Editorenordner')
        bytes = ods.tobytes()

        return bytes

    def export_to_xls(self, filename, editor):
        import xlsxwriter
        from database.file_formats.exporter.monodi.ods import MonodiXlsxConfig
        from io import BytesIO

        output = BytesIO()

        workbook = xlsxwriter.Workbook(output)
        config = MonodiXlsxConfig()
        worksheet = workbook.add_worksheet()

        for x in config.entries:
            worksheet.write(x.cell.row, x.cell.column, x.value)

        worksheet.write(1, config.dict['Textinitium Editionseinheit'].cell.column, self.textinitium)
        worksheet.write(1, config.dict['Startseite'].cell.column, self.start.page_name)
        worksheet.write(1, config.dict['Startzeile'].cell.column, self.start.row)
        worksheet.write(1, config.dict['Endseite'].cell.column, self.end.page_name)
        worksheet.write(1, config.dict['Endzeile'].cell.column, self.end.row)
        worksheet.write(1, config.dict['Editor'].cell.column, str(editor))
        worksheet.write(1, config.dict['Doc-Id\' (intern)'].cell.column, self.monody_id)
        worksheet.write(1, config.dict['Quellen-ID (intern)'].cell.column, 'Editorenordner')
        workbook.close()
        xlsx_data_bytes = output.getvalue()
        return xlsx_data_bytes

    def get_page_line_of_document(self, book) -> List[Tuple[Line, DatabasePage]]:
        line_page_pair = []
        started = False
        pages = [DatabasePage(book, x) for x in self.pages_names]
        for page in pages:
            for line in page.pcgts().page.reading_order.reading_order:
                if page.pcgts().page.p_id == self.end.page_id:
                    if line.id == self.end.line_id:
                        break
                if line.id == self.start.line_id or started:
                    started = True
                    line_page_pair.append((line, page))
            else:
                continue
            break
        return line_page_pair

    def update_textline_count(self, book: DatabaseBook):
        self.textline_count = len(self.get_page_line_of_document(book))

    def get_text_list_of_line_document(self, book):
        line_text = self.get_page_line_of_document(book)

        line_text = [i[0].text() for i in line_text]

        return line_text

    def get_text_of_document(self, book):
        line_text = self.get_text_list_of_line_document(book)
        text = " ".join(line_text)
        return text

    def get_text_of_document_by_line(self, book, index):
        line_text = self.get_page_line_of_document(book)
        return line_text[int(index)][0].text()

    def get_image_of_document_by_line(self, book, index):
        lines = self.get_page_line_of_document(book)
        line, page = lines[int(index)]
        page: DatabasePage = page
        image = Image.open(page.file('color_highres_preproc').local_path())
        coords = line.coords
        coords = page.pcgts().page.page_to_image_scale(coords, PageScaleReference.HIGHRES)

        image = coords.extract_from_image(np.array(image))
        return image

    def update_pcgts(self, book, lines):
        pcgts_to_save = []
        line_page_pairs = self.get_page_line_of_document(book=book)
        for line, line_page_pair in zip(lines["lines"], line_page_pairs):
            line_page_pair[0].sentence = Sentence.from_string(line["gt"])
            pcgts_to_save.append(line_page_pair[1])
        for i in set(pcgts_to_save):
            pcgts = i.pcgts()
            pcgts.page.annotations.connections.clear()
            pcgts.to_file(i.file('pcgts').local_path())

    def get_book_u_id(self, book_str=""):
        return self.start.page_name + "-" + str(self.start.row) + "-" + str(self.end.row)

    def get_symbols(self, book) -> Tuple[List[List[MusicSymbol]], List[LineMetaInfos]]:

        lines = self.get_page_line_of_document(book)
        symbols = []
        meta = []
        for line, page in lines:
            page: DatabasePage = page
            print(page.page)
            music_line = page.pcgts().page.closest_music_line_to_text_line(line)
            if music_line is None:
                continue
            text_lines_of_page = page.pcgts().page.all_text_lines(True)
            m_lines = [i for i in text_lines_of_page if  page.pcgts().page.closest_music_line_to_text_line(i) and
                       page.pcgts().page.closest_music_line_to_text_line(i).id == music_line.id]
            lefts_text_lines = sorted([i.coords.aabb().left() for i in m_lines])
            if len(m_lines) > 1:
                left = line.coords.aabb().left()
                right = line.coords.aabb().right()
                l_p = -1 if lefts_text_lines.index(left) == 0 else left + 0.03
                r_p = 1 if lefts_text_lines.index(left) == len(lefts_text_lines) else right + 0.03
                filtered_symbols = [i for i in music_line.symbols if l_p < i.coord.x < r_p]
                symbols.append(filtered_symbols), meta.append(LineMetaInfos(music_line.id, l_p, r_p, page.page))

            else:
                symbols.append(music_line.symbols), meta.append(LineMetaInfos(music_line.id, -1, -1, page.page))
        return symbols, meta

    def get_symbol_of_page(self, book):
        pages = [DatabasePage(book, x) for x in self.pages_names]

        symbols = []
        for page in pages:
            symbols.append(page.pcgts().page.get_all_music_symbols_of_page())
        return symbols

