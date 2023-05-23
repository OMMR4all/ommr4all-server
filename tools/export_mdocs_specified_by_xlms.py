import json

import openpyxl
from openpyxl import Workbook

from database import DatabaseBook, DatabasePage, DatabaseFile
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter

if __name__ == "__main__":
    book = DatabaseBook('mul_2_rsync_gt')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    wb_obj = openpyxl.load_workbook("/tmp/CM Default Metadatendatei.xlsx")
    sheet_obj = wb_obj.active
    #cell_obj = sheet_obj.cell(row=1, column=30)
    #print(cell_obj.value)
    i = 2
    while True:
        cell_obj = sheet_obj.cell(row=i, column=30).value
        if cell_obj:
            try:
                document = documents.database_documents.get_document_by_monodi_id(cell_obj)
                pages = [DatabasePage(book, x) for x in document.pages_names]
                pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
                root = PcgtsToMonodiConverter(pcgts, document=document)
                json_data = root.get_Monodi_json(document=document, editor=str("OMMR4all"))
                with open(f"/tmp/export/{document.monody_id}.json", "w") as f:
                    json.dump(json_data, f, indent=4)
            except:
                print(1)
            pass

        else:
            break
        i = i+1

    #docs_ids = []

    #pages = [DatabasePage(book, x) for x in document.pages_names]
    #pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in
    #         pages]
    #root = PcgtsToMonodiConverter(pcgts, document=document)
    #json_data = root.get_Monodi_json(document=document, editor=str(editor))
