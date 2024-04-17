import json
import os

import edlib
import openpyxl

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from tools.rf.transfer_ocr_texts_to_regios import lyric_order
from tools.simple_gregorianik_text_export import Lyric_info

"45"
"CM Default Metadatendatei_count.xlsx"
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

if __name__ == "__main__":
    dataset_json_file_path = "/home/alexanderh/Documents/datasets/rendered_files2/"

    rendered_json_files = {}
    for file in os.listdir(dataset_json_file_path):
        if file.endswith(".json"):
            rendered_json_files[file.replace(".json", "")] = file
    workbook = openpyxl.load_workbook('/tmp/CM Default Metadatendatei.xlsx')
    sheet = workbook.active
    book = DatabaseBook('Graduel_Syn22_03_24')
    documents = DatabaseBookDocuments().load(book)
    current_page = None
    updates = {}
    for i in range(3, 2500):
        page = sheet["O" + str(i)].value
        if current_page != page:
            current_page = page
            if len(updates) > 0:
                main_var = "Kl"
                val = updates.get(main_var, None)
                if val is not None:
                    doc = documents.database_documents.get_document_by_id(sheet["A" + str(val)].value)
                    text = doc.get_text_of_document(book).replace(".", "").replace("-", "").replace(" ", "")
                    file_path = dataset_json_file_path + rendered_json_files[doc.start.page_name]
                    with open(file_path, 'r', encoding='utf-8') as f:
                        t = json.load(f)
                        lyric_info = Lyric_info.from_dict(t)
                        latine = lyric_info.latine.replace(".", "")

                    variants = {}

                    for s in lyric_order:
                        for t in lyric_info.variants:
                            if t.source == s:
                                variants[s] = t.latine
                                break
                    print(text.replace(".", "").replace("-", "").replace(" ", ""))
                    print(variants[main_var].replace(".", "").replace("-", "").replace(" ", ""))

                    for key, value in updates.items():

                        doc = documents.database_documents.get_document_by_id(sheet["A" + str(value)].value)
                        text_var = doc.get_text_of_document(book).replace(".", "").replace("-", "").replace(" ", "")
                        ed = edlib.align(text_var, text, mode="NW",
                                         task="path")
                        if ed["editDistance"] < 5:

                            if ed["editDistance"] == 0:
                                sheet["AT" + str(value)] = "no"

                            else:
                                sheet["AT" + str(value)] = "no1"

                        else:
                            sheet["AT" + str(value)] = "yes"

                        ed = edlib.align(text_var, variants[key].replace(".", "").replace("-", "").replace(" ", ""), mode="NW",
                                         task="path")
                        if ed["editDistance"] < 5:

                            if ed["editDistance"] == 0:
                                sheet["AM" + str(value)] = "no"
                            else:
                                sheet["AM" + str(value)] = "no1"
                        else:
                            sheet["AM" + str(value)] = "yes"

            updates = {}
            print("___")
        manuscript = sheet["AH" + str(i)].value
        if manuscript is None:
            continue
        updates[manuscript] = i
        print(sheet["AD" + str(i)].value)
        doc = documents.database_documents.get_document_by_id(sheet["A" + str(i)].value)
        if doc:
            print(doc.start)
        print(sheet["AM" + str(i)].value)
    workbook.save('/tmp/Graduale.xlsx')