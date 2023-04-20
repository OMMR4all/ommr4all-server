import json
from collections import defaultdict

from tools.evaluated_monodi_texts import populate
from collections import Counter

export_dir = "/home/alexanderh/Downloads/mc_export/export/"
monodoy_documents, lyrics, t_with_syllabels = populate("/home/alexanderh/Downloads/mc_export/export/")

syllable_dict = defaultdict(list)


def to_string(syllable_text):
    s = ""
    for i in syllable_text:
        if len(i) > 0:
            if i[-1] == "-":
                s += i
            else:
                s += i + " "
    return s


for i in t_with_syllabels:
    words = to_string(i).split(" ")
    for word in words:
        w_syl = word.replace("-", "").lower()
        syllable_dict[w_syl].append(word.lower())

syllables = {}

for i in sorted(syllable_dict.keys()):
    alternatives = syllable_dict[i]
    c = Counter(alternatives)
    hyph = ""
    count = 0
    for t in c.keys():
        if c[t] > count:
            hyph = t
            count = c[t]
            print(f'word {i}, hyphenation: {t}, count: {c[t]}')
    if count > 1:
        syllables[i] = hyph

with open("syllable_dictionary.json", "w", encoding='utf-8') as file:
    json.dump(syllables, file, ensure_ascii=False, indent=4)


