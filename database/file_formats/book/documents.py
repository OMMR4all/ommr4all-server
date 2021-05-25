import json
from datetime import datetime
from typing import List

from database.file_formats.book.document import Document


class Documents:
    def __init__(self, documents: List[Document]):
        self.documents: List[Document] = documents

    @staticmethod
    def from_json(json: dict):
        if type(json) == dict:
            return Documents(
                documents=[Document.from_json(l) for l in json.get('documents', [])]
            )
        else:
            return None

    def to_json(self):
        return {
            "documents": [document.to_json() for document in self.documents] if self.documents else [],
        }

    def get_document_by_id(self, id):
        for x in self.documents:
            if x.doc_id == id:
                return x
        return None

    @staticmethod
    def export_documents_to_ods(documents: List[Document], filename, username):
        from database.file_formats.exporter.monodi.ods import MonodiOdsConfig
        from ezodf import newdoc, Paragraph, Heading, Sheet
        import ezodf
        ods = newdoc(doctype='ods', filename=filename)
        config = MonodiOdsConfig()
        sheet = ezodf.Sheet('Tabellenblatt1', size=(len(documents) + 1, config.length))
        ods.sheets += sheet
        for x in config.entries:
            sheet[x.cell.get_entry()].set_value(x.value)
        for doc_ind, doc in enumerate(documents, start=2):
            sheet[''.join([config.dict['Textinitium Editionseinheit'].cell.column, str(doc_ind)])].set_value(doc.textinitium)
            sheet[''.join([config.dict['Startseite'].cell.column, str(doc_ind)])].set_value( doc.start.page_name)
            sheet[''.join([config.dict['Startzeile'].cell.column, str(doc_ind)])].set_value( doc.start.row)
            sheet[''.join([config.dict['Endseite'].cell.column, str(doc_ind)])].set_value( doc.end.page_name)
            sheet[''.join([config.dict['Endzeile'].cell.column, str(doc_ind)])].set_value( doc.end.row)
            sheet[''.join([config.dict['Editor'].cell.column, str(doc_ind)])].set_value(username)
            sheet[''.join([config.dict['Doc-Id\' (intern)'].cell.column, str(doc_ind)])].set_value(doc.monody_id)
            sheet[''.join([config.dict['Quellen-ID (intern)'].cell.column, str(doc_ind)])].set_value('Editorenordner')
        bytes = ods.tobytes()

        return bytes

    @staticmethod
    def export_documents_to_xls(documents: List[Document], filename, editor):
        import xlsxwriter
        from database.file_formats.exporter.monodi.ods import MonodiXlsxConfig
        from io import BytesIO

        output = BytesIO()

        workbook = xlsxwriter.Workbook(output)
        config = MonodiXlsxConfig()
        worksheet = workbook.add_worksheet()

        for x in config.entries:
            worksheet.write(x.cell.row, x.cell.column, x.value)
        for doc_ind, doc in enumerate(documents, start=1):
            worksheet.write(doc_ind, config.dict['Textinitium Editionseinheit'].cell.column, doc.textinitium)
            worksheet.write(doc_ind, config.dict['Startseite'].cell.column, doc.start.page_name)
            worksheet.write(doc_ind, config.dict['Startzeile'].cell.column, doc.start.row)
            worksheet.write(doc_ind, config.dict['Endseite'].cell.column, doc.end.page_name)
            worksheet.write(doc_ind, config.dict['Endzeile'].cell.column, doc.end.row)
            worksheet.write(doc_ind, config.dict['Editor'].cell.column, str(editor))
            worksheet.write(doc_ind, config.dict['Doc-Id\' (intern)'].cell.column, doc.monody_id)
            worksheet.write(doc_ind, config.dict['Quellen-ID (intern)'].cell.column, 'Editorenordner')
        workbook.close()
        xlsx_data_bytes = output.getvalue()
        return xlsx_data_bytes

if __name__ == '__main__':
    documents = Documents([Document("1", ["page1", "page2"], 1000), Document("2", ["page2", "page3"], 1002)])
    js = json.dumps(documents.to_json(), indent=2)
    documents = Documents.from_json(json.loads(js))
    print(documents.to_json())
    # print(json.dumps(s.to_json(), indent=2))
