import json
import os

import openpyxl

from database import DatabaseBook, DatabasePage, DatabaseFile
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter

def soruce_meta(id):
    meta = {
        "id": id,
        "quellensigle": "",
        "herkunftsregion": "",
        "herkunftsort": "",
        "herkunftsinstitution": "",
        "ordenstradition": "",
        "quellentyp": "",
        "bibliotheksort": "",
        "bibliothek": "",
        "bibliothekssignatur": "",
        "kommentar": "",
        "datierung": "",
        "status": "",
        "jahrhundert": "",
        "manifest": "",
        "foliooffset": 0,
        "publish": None
    }
    return meta

if __name__ == "__main__":
    source = "Graduale_Synopticum_22_03_24"
    source_meta = soruce_meta(source)
    book = DatabaseBook('Graduel_Syn22_03_24')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    wb_obj = openpyxl.load_workbook("/tmp/Graduale_stats.xlsx", data_only=True)
    sheet_obj = wb_obj.active
    #cell_obj = sheet_obj.cell(row=1, column=30)
    #print(cell_obj.value)
    i = 3
    os.mkdir("/tmp/export/")
    os.mkdir(f"/tmp/export/{source}")
    with open(f"/tmp/export/{source}/meta.json", "w") as f:
        json.dump(source_meta, f, indent=4)
    while True:
        cell_obj = sheet_obj["A" + str(i)].value
        skip = sheet_obj["BJ" + str(i)].value
        #print(skip)
        #print(sheet_obj["O" + str(i)].value)
        if skip == "yes":
            print(sheet_obj["O" + str(i)].value)
            i = i + 1
            continue

        if cell_obj:
            try:
                document = documents.database_documents.get_document_by_id(cell_obj)
                pages = [DatabasePage(book, x) for x in document.pages_names]
                pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
                if i == 9:
                    pass
                    pass
                root = PcgtsToMonodiConverter(pcgts, document=document, replace_filename="folio")
                meta, nodes = root.get_meta_and_notes(document=document, editor=str("ANONYMIZED"), sourceIIF="Graduale_Synopticum",
                           doc_source="Graduale Synopticum", suffix=".png", url="ANONYMIZED.xyz/iiif/3/")
                os.mkdir(f"/tmp/export/{source}/{document.monody_id}")
                with open(f"/tmp/export/{source}/{document.monody_id}/data.json", "w") as f:
                    json.dump(nodes, f, indent=4)
                with open(f"/tmp/export/{source}/{document.monody_id}/meta.json", "w") as f:
                    json.dump(meta, f, indent=4)
            except Exception as e:
                pass
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
