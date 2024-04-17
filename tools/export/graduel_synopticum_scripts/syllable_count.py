import json
import os

import edlib
import openpyxl

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts import Line


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
    workbook = openpyxl.load_workbook('/tmp/Graduale_Results_05_04_23.xlsx')
    sheet = workbook.get_sheet_by_name("Sheet1")
    book = DatabaseBook('Graduel_Syn22_03_24')
    documents = DatabaseBookDocuments().load(book)
    book2 = DatabaseBook('Graduel_Syn22_03_24_pred')
    documents2 = DatabaseBookDocuments().load(book2)

    current_page = None
    updates = {}
    for i in range(3, 2500):
        doc = documents.database_documents.get_document_by_id(sheet["A" + str(i)].value)
        if not doc:
            sheet["AV" + str(i)].value = "yes"
            continue
        lines = doc.get_page_line_of_document(book)
        syls1 = 0
        syls2 = 0
        for t in lines:
            line: Line = t[0]
            syls1 += len(line.sentence.syllables)
        doc2 = documents2.database_documents.get_document_by_id(sheet["A" + str(i)].value)
        if not doc2:
            sheet["AV" + str(i)].value = "yes"
            continue
        lines2 = doc2.get_page_line_of_document(book2)
        for t in lines2:
            line: Line = t[0]
            syls2 += len(line.sentence.syllables)
        if syls1 != syls2:
            sheet["AV" + str(i)].value = "yes"
        else:
            sheet["AV" + str(i)].value = "no"
    workbook.save('/tmp/Graduale.xlsx')