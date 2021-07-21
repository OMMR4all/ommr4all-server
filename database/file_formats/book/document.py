from datetime import datetime
from typing import List
import uuid

import ezodf

from database import DatabasePage


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
                 monody_id=None, doc_id=None, textinitium=''):
        self.monody_id = monody_id if monody_id else str(uuid.uuid4())
        self.doc_id = doc_id if doc_id else str(uuid.uuid4())
        self.pages_ids: List[int] = page_ids
        self.pages_names: List[str] = page_names
        self.start: DocumentConnection = start
        self.end: DocumentConnection = end
        self.textinitium = textinitium

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

        }

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

    def get_text_of_document(self, book):
        text = ""
        started = False
        pages = [DatabasePage(book, x) for x in self.pages_names]
        for page in pages:
            for line in page.pcgts().page.all_text_lines():

                if page.pcgts().page.p_id == self.end.page_id:
                    if line.id == self.end.line_id:
                        break
                if line.id == self.start.line_id or started:
                    started = True
                    text += line.text() + " "
            else:
                continue
            break
        return text
