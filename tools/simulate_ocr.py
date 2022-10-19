import json
import random

import edlib
import pandas as pd
import xlsxwriter
from tqdm import tqdm

from tools.evaluated_monodi_texts import populate
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics, Lyric_info


def update_index(index_list: list, index, operation: str):
    for ind, i in enumerate(index_list):
        if i >= index:
            if operation == "insert":
                index_list[ind] = i + 1
            else:
                index_list[ind] = i - 1
    if operation == "insert":
        index_list.append(index)
    return index_list


def simulate_ocr_error(lyric: str, ocr_accuracy: float = 0.9):
    chars = "abcdefghijklmnopqrstuvwxyz "
    len_lyric = len(lyric)
    nb_of_errors = round((1 - ocr_accuracy) * len_lyric)
    new_error_string = lyric
    operations = ["insert", "delete"]
    index_of_errors = []
    for i in range(nb_of_errors):
        operation = random.choice(operations)
        index = random.randint(0, len(new_error_string) - 1)
        while index in index_of_errors:
            index = random.randint(0, len(new_error_string) - 1)
        update_index(index_of_errors, index, operation)
        if operation == "insert":
            char = random.choice(chars)
            new_error_string = new_error_string[:index] + char + new_error_string[index:]
        elif operation == "delete":
            new_error_string = new_error_string[:index] + new_error_string[index + 1:]
    return new_error_string

class OCRSimEntry:
    monody: str
    most_similar_text: str
    sim_score: float
    simulated_ocr: str

if __name__ == "__main__":

    #export_dir = "/home/alexanderh/Downloads/mc_export/export/"
    #monodoy_documents = populate("/home/alexanderh/Downloads/mc_export/export/")
    #print("loaded monody")
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




    error = simulate_ocr_error("ich laufe hier herum. Herum laufe ich hier")
    path = "percentage_of_monodicum_included_in_latine_collection_sub_string_extended_with_secondary_similarity_score.ods"
    content = pd.read_excel(path, engine='odf', skiprows=2)
    #print(content)
    #print(type(content))
    #print(len(list(content.iterrows())))
    scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sim_count = 100
    for score in tqdm(tqdm(scores)):
        sim_count_now = 0
        workbook = xlsxwriter.Workbook('simulated_ocr_eval_score_{}_big_db.xlsx'.format(score))
        worksheet = workbook.add_worksheet()
        worksheet.write(1, 0, "monodi_text")
        worksheet.write(1, 1, "Most_similiar_text_in_db")
        worksheet.write(1, 2, "Sim_score_between_cm_and_db_text")
        worksheet.write(1, 3, "Simulated_OCR_Text")
        worksheet.write(1, 4, "Most_similar_text_to_ocr")
        worksheet.write(1, 5, "edit distance between text of col 2 and text of col 5")

        row_i = 3

        for i in tqdm(tqdm(content.iterrows())):
            row = i[1].values.flatten().tolist()
            monodi_text = row[0]
            db_text = row[1]
            sim_score = row[9]
            if sim_score > 0.9:
                sim_count_now += 1
                simulate_ocr_error_text = simulate_ocr_error(monodi_text, score)
                lowest_ed = 99999999999999999999999999999999
                lowest_text = ""
                for b in lyrics.lyrics:
                    b: Lyric_info = b
                    text2 = b.latine.lower()
                    ed = edlib.align(simulate_ocr_error_text, text2, mode="SHW")
                    if ed["editDistance"] < lowest_ed:
                        lowest_ed = ed["editDistance"]
                        lowest_text = text2
                ed2 = edlib.align(db_text, lowest_text, mode="SHW")

                worksheet.write(row_i, 0, monodi_text)
                worksheet.write(row_i, 1, db_text)
                worksheet.write(row_i, 2, sim_score)
                worksheet.write(row_i, 3, simulate_ocr_error_text)
                worksheet.write(row_i, 4, lowest_text)
                worksheet.write(row_i, 5, ed2["editDistance"])
                row_i += 1
        workbook.close()

