import json
from collections import defaultdict

import edlib
import xlsxwriter
from tqdm import tqdm

from tools.evaluated_monodi_texts import populate
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics, Lyric_info, Variant

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

# lyrics = Lyrics(lyrics.lyrics + lyrics2.lyrics + lyrics3.lyrics)

workbook = xlsxwriter.Workbook('lyrics_by_source_new.xlsx')
worksheet = workbook.add_worksheet()
row = 3
monody_texts = [(monodoy_documents[i][1].lower().replace("<", " ").replace(">", " "), "cm") for i in monodoy_documents.keys()]
latine_gr_texts = [(x.latine.lower().replace("<", " ").replace(">", " "), "gr") for x in lyrics.lyrics]
latine_an_texts = [(x.latine.lower().replace("<", " ").replace(">", " "), "an") for x in lyrics2.lyrics]
cantus_texts = [(x.latine.lower().replace("<", " ").replace(">", " "), "cantus") for x in lyrics3.lyrics]
print("cm: {}, gr:{}, an:{}, cn:{}".format(len(monody_texts), len(latine_gr_texts), len(latine_an_texts), len(cantus_texts)))
all_texts = monody_texts + latine_gr_texts + latine_an_texts + cantus_texts
all_texts = sorted(set(all_texts), key=lambda x: x[0])
# texts = defaultdict(list)
# for i in monodoy_documents.keys():
#    initial, text = monodoy_documents[i]
#    texts[text].append(i)
similar_docs_monody = defaultdict(list)
similarity_score = 0.9
for i in tqdm(tqdm(all_texts)):
    text, source = i
    if len(text) == 0:
        continue
    insert = True
    for text2 in similar_docs_monody.keys():

        # initial2, text2 = monodoy_documents[t]
        ed = edlib.align(text, text2)
        sim_score = 1 - (ed["editDistance"] / len(text))
        if sim_score >= similarity_score:
            similar_docs_monody[text2].append((text, source))
            insert = False
    if insert:
        similar_docs_monody[text].append((text, source))

lyrics_by_sources = []
for i in sorted(similar_docs_monody.keys()):
    text = i
    sim = similar_docs_monody[i]
    variant_texts = "\n ".join(set(["Lyric:{} |id:{}".format(x[0], x[1]) for x in sim])).strip()
    variants = []
    for a in sim:
        variants.append(Variant(source=a[1], latine=a[0]))
    worksheet.write(row, 0, text)
    worksheet.write(row, 1, sim[0][1])
    worksheet.write(row, 2, len(sim))
    worksheet.write(row, 3, variant_texts)
    row += 1
    lyrics_by_sources.append(Lyric_info(index=None, id = None, latine = i, meta_info=None, meta_infos_extended=None, variants=variants))

lyrics_by_sources = Lyrics(lyrics_by_sources)
with open('lyrics_by_sources.json', 'w', encoding='utf-8') as f:
    json.dump(lyrics_by_sources.to_dict(), f, ensure_ascii=False, indent=4)

workbook.close()

