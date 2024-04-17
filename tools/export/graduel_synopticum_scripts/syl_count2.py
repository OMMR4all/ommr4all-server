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

    workbook = openpyxl.load_workbook('/tmp/Graduale_stats.xlsx')
    sheet = workbook.get_sheet_by_name("Sheet1")
    sheet2 = workbook.get_sheet_by_name("Symbols Doc Statistics1")


    current_page = None
    updates = {}
    for i in range(3, 2500):
        c_page = sheet["O" + str(i)].value

        if current_page != c_page:
            kl_stats = updates.get("Kl", None)
            if kl_stats:
                for t in updates.items():
                    sheet["BB" + str(t[1]["index"])].value = "yes" if t[1]["syl_stats"] != kl_stats["syl_stats"] else "no"
            updates = {}

        c_manuscript = sheet["AH" + str(i)].value
        c_syl_stats = sheet2["F" + str(i)].value
        updates[c_manuscript] = {"page": c_page, "index": i, "syl_stats": c_syl_stats}
        current_page = c_page
    workbook.save('/tmp/Graduale.xlsx')