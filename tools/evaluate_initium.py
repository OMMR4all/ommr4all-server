import json
import os
from tqdm import tqdm
from database.file_formats.importer.mondodi.simple_import import simple_monodi_data_importer2
from omr.text_matching.populate_db import MonodiImportStructure
from tools.word_frequency_list import list_dirs
import edlib
def to_string(syllable_text):
    s = ""
    for i in syllable_text:
        if len(i) > 0:
            if i[-1] == "-":
                s += i[:-1]
            else:
                s += i + " "
    return s
def populate(path):
    documents_memory_db = {}
    documents = documents_gen(path)
    for i in tqdm(tqdm(documents)):
        if os.path.exists(i.data):
            initium = generate_initiums(i)
            if len(initium) > 0:
                gt_text = [s.syllable.text for s in initium[0].neumes]
                #print(to_string(gt_text) + " " + i.document_id)
                documents_memory_db[i] = [to_string(gt_text), to_string([t.syllable.text for s in initium for t in s.neumes])]

    import xlsxwriter
    workbook = xlsxwriter.Workbook('Similar_documents.xlsx')
    worksheet = workbook.add_worksheet()
    row = 3
    col = 0
    for key in tqdm(tqdm(documents_memory_db.keys())):
        item = documents_memory_db[key][0]
        text1 = documents_memory_db[key][1]
        if len(item) > 0:
            lowest_ed = 99999
            text = ""
            lw_item2 = None
            lw_text2 = ""
            for key2 in documents_memory_db.keys():
                if key.data != key2.data:
                    item2 = documents_memory_db[key2][0]
                    text2 = documents_memory_db[key2][1]
                    ed = edlib.align(item, item2)
                    if ed["editDistance"] < lowest_ed:
                        lowest_ed = ed["editDistance"]
                        text = item2
                        lw_item2 = key2
                        lw_text2 = text2

            worksheet.write(row, 0, item)
            worksheet.write(row, 1, text)
            worksheet.write(row, 2, lowest_ed)
            worksheet.write(row, 3, len(text1))
            worksheet.write(row, 4, len(lw_text2))
            worksheet.write(row, 5, "/".join(key.data.split("/")[6:]))
            "/home/alexanderh/Downloads/mc_export/export/AugW 13/27ddd32c-1692-40c1-9255-e27b278bc657/data.json"
            worksheet.write(row, 6, "/".join(lw_item2.data.split("/")[6:]) if lw_item2 else "")
            worksheet.write(row, 7, text1)
            worksheet.write(row, 8, lw_text2)
            row += 1
    workbook.close()

        #edlibAlign("hello", 5, "world!", 6, edlibDefaultAlignConfig()).editDistance;



# print(json.loads(self.settings.params.documentText))
def documents_gen(export_dir):
    from ommr4all.settings import BASE_DIR

    dir = list_dirs(export_dir, True)
    for x in dir:
        s_dir = os.path.join(export_dir, x)
        docs = list_dirs(s_dir, True)
        source = os.path.join(s_dir, "meta.json")

        for doc in docs:
            d_dir = os.path.join(s_dir, doc)
            data = os.path.join(d_dir, "data.json")
            meta = os.path.join(d_dir, "meta.json")
            yield MonodiImportStructure(source, data, meta, doc)


def generate_initiums(path):
    with open(path.data) as json_data:
        json_string = json.load(json_data)
        abc = simple_monodi_data_importer2(json_string, ignore_liquescent=True)
    return abc


if __name__ == "__main__":
    export_file = "/home/alexanderh/Downloads/mc_export/export/"
    populate(export_file)
    #a = MonodiImportStructure("", "/home/alexanderh/Downloads/mc_export/export/Ba 12/bec670cd-1045-4cd1-b809-a304ea6afc75/data.json" ," ", " ")
    #ab = generate_initiums(a)
    #strin = to_string([t.syllable.text for s in ab for t in s.neumes])
    #print(ab)
    #print(strin)
    pass
