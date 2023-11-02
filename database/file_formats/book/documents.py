import json
from datetime import datetime
from typing import List

from database.file_formats.book.document import Document, DocumentMetaInfos
from database.file_formats.pcgts import Page
from tools.simple_gregorianik_text_export import Lyric_info


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

    def update_document_meta_infos(self, m_infos: Lyric_info, doc_uuid):
        for i in self.documents:
            if i.doc_id == doc_uuid:
                festum = ""
                d = m_infos
                if m_infos:
                    if m_infos.meta_infos_extended:
                        if len(m_infos.meta_infos_extended) > 0:
                            if m_infos.meta_infos_extended[0].festum is not None:
                                festum += m_infos.meta_infos_extended[0].festum
                            if m_infos.meta_infos_extended[0].dies is not None:
                                festum += m_infos.meta_infos_extended[0].dies

                    d = DocumentMetaInfos(cantus_id=m_infos.cantus_id, initium=m_infos.initium, genre=m_infos.genre,
                                          url=m_infos.url, dataset_source=m_infos.dataset_source, festum=festum, extended_source=m_infos.source)
                    i.textinitium = m_infos.initium
                i.document_meta_infos = d
                return
        pass

    def get_document_by_id(self, id):
        for x in self.documents:
            if x.doc_id == id:
                return x
        return None

    def get_document_by_monodi_id(self, id):
        for x in self.documents:
            if x.monody_id == id:
                return x
        return None

    def get_document_by_b_uid(self, b_uid):
        for x in self.documents:
            if x.get_book_u_id() == b_uid:
                return x
        return None

    def get_documents_of_page(self, page: Page):
        docs: List[Document] = []
        for x in self.documents:
            if x.start.page_id == page.p_id:
                docs.append(x)
        return docs

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
            sheet[''.join([config.dict['Textinitium Editionseinheit'].cell.column, str(doc_ind)])].set_value(
                doc.textinitium)
            sheet[''.join([config.dict['Startseite'].cell.column, str(doc_ind)])].set_value(doc.start.page_name)
            sheet[''.join([config.dict['Startzeile'].cell.column, str(doc_ind)])].set_value(doc.start.row)
            sheet[''.join([config.dict['Endseite'].cell.column, str(doc_ind)])].set_value(doc.end.page_name)
            sheet[''.join([config.dict['Endzeile'].cell.column, str(doc_ind)])].set_value(doc.end.row)
            sheet[''.join([config.dict['Editor'].cell.column, str(doc_ind)])].set_value(username)
            sheet[''.join([config.dict['Doc-Id\' (intern)'].cell.column, str(doc_ind)])].set_value(doc.monody_id)
            sheet[''.join([config.dict['Quellen-ID (intern)'].cell.column, str(doc_ind)])].set_value('Editorenordner')
        bytes = ods.tobytes()

        return bytes

    @staticmethod
    def export_documents_to_xls(documents: List[Document], filename, editor, book=None):
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
            if doc.document_meta_infos:
                worksheet.write(doc_ind, config.dict['Gattung1'].cell.column, doc.document_meta_infos.genre)
                worksheet.write(doc_ind, config.dict['Fest'].cell.column, doc.document_meta_infos.festum)
                worksheet.write(doc_ind, config.dict['Verlinkung'].cell.column, doc.document_meta_infos.url)

            if doc.document_meta_infos and doc.document_meta_infos.initium and len(doc.document_meta_infos.initium) > 0:
                worksheet.write(doc_ind, config.dict['Textinitium Editionseinheit'].cell.column, doc.document_meta_infos.initium)
            else:
                if  doc.textinitium:
                    worksheet.write(doc_ind, config.dict['Textinitium Editionseinheit'].cell.column, doc.textinitium.replace("-", ""))
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
