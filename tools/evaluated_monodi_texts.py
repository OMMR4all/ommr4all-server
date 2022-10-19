import json
import os
from collections import defaultdict

import edlib
import xlsxwriter
from tqdm import tqdm

from tools.evaluate_initium import documents_gen, generate_initiums, to_string
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics, Lyric_info


def populate(path):
    documents_memory_db = {}
    documents = documents_gen(path)
    for i in tqdm(tqdm(documents)):
        if os.path.exists(i.data):
            initium = generate_initiums(i)
            if len(initium) > 0:
                gt_text = [s.syllable.text for s in initium[0].neumes]
                # print(to_string(gt_text) + " " + i.document_id)
                documents_memory_db[i] = [to_string(gt_text),
                                          to_string([t.syllable.text for s in initium for t in s.neumes])]

    return documents_memory_db


if __name__ == "__main__":
    export_dir = "/home/alexanderh/Downloads/mc_export/export/"
    monodoy_documents = populate("/home/alexanderh/Downloads/mc_export/export/")
    print("loaded monody")
    with open("latine_collection_gr.json") as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    print("loaded latine gr")
    with open("latine_collection_an.json") as f:
        json2 = json.load(f)
        lyrics2 = Lyrics.from_dict(json2)
    print("loaded latine an")

    lyrics3 = get_csv_text("/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/chant.csv")
    print("loaded cantus csv")

    lyrics = Lyrics(lyrics.lyrics + lyrics2.lyrics + lyrics3.lyrics)
    workbook = xlsxwriter.Workbook('percentage_of_monodicum_included_in_latine_collection_sub_string_extended.xlsx')
    worksheet = workbook.add_worksheet()
    row = 3
    #texts = defaultdict(list)
    #for i in monodoy_documents.keys():
    #    initial, text = monodoy_documents[i]
    #    texts[text].append(i)
    similar_docs_monody = defaultdict(list)
    similarity_score = 0.9
    for i in tqdm(tqdm(monodoy_documents.keys())):
        initial, text = monodoy_documents[i]
        if len(text) == 0:
            continue
        insert = True
        for t in similar_docs_monody.keys():
            initial2, text2 = monodoy_documents[t]
            ed = edlib.align(text, text2)
            sim_score = 1 - (ed["editDistance"] / len(text))
            if sim_score >= similarity_score:
                similar_docs_monody[t].append(i)
                insert = False
        if insert:
            similar_docs_monody[i].append(i)
    print(len(similar_docs_monody.keys()))
    for i in tqdm(tqdm(similar_docs_monody.keys())):
        #initial, text = monodoy_documents[i]
        docs = similar_docs_monody[i]
        variant_texts = "\n ".join(set([monodoy_documents[x][1] for x in docs])).strip()

        text = monodoy_documents[i][1]
        nb = len(docs)
        docs = "; ".join([x.data[len(export_dir):-10] for x in docs])
        text = text.lower().replace("<", " ").replace(">", " ")
        lowest_ed = 9999999999999999999999999
        lowest_text = ""
        for b in lyrics.lyrics:
            b: Lyric_info = b
            text2 = b.latine.lower()
            ed = edlib.align(text, text2, mode="SHW")
            if ed["editDistance"] < lowest_ed:
                lowest_ed = ed["editDistance"]
                lowest_text = text2
        shorter_text = text if len(text) < len(lowest_text) else lowest_text
        longer_text = text if len(text) > len(lowest_text) else lowest_text
        ed = edlib.align(shorter_text, longer_text, mode="SHW")
        edit_distance = ed["editDistance"]
        worksheet.write(row, 0, text)
        worksheet.write(row, 1, lowest_text)
        worksheet.write(row, 2, lowest_ed)
        worksheet.write(row, 3, edit_distance)
        worksheet.write(row, 4, nb)
        worksheet.write(row, 5, str(docs))
        worksheet.write(row, 6, variant_texts)

        row += 1

    workbook.close()
