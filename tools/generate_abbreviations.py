import csv
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Acronym:
    acronym: str
    count: int


@dataclass_json
@dataclass
class Abbreviation:
    full_word: str
    acronym: List[Acronym]


@dataclass_json
@dataclass
class Abbreviations:
    abbreviations: List[Abbreviation]


def get_abbreviations(csv_path):
    dd = defaultdict(list)
    chants = []
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            chant = row[18]
            chant = chant.replace("-", "").replace("*", "").replace("(", "").replace(")", "").replace(".",
                                                                                                      "").strip().lower().replace(
                "{", "").replace("}", "").replace("[", "").replace("]", "").replace(",", "").replace("#", "").replace(
                "|", "").replace(":", "")
            chant_manuscript = row[19]
            chant_manuscript = chant_manuscript.replace("-", "").replace("*", "").replace("(", "").replace(")",
                                                                                                           "").replace(
                ".", "").strip().lower().replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace(
                ",", "").replace("#", "").replace("|", "").replace(":", "")

            if chant != "" and chant_manuscript != "":
                if len(chant.split(" ")) == len(chant_manuscript.split(" ")):
                    split1 = chant.split(" ")
                    split2 = chant_manuscript.split(" ")
                    for i in range(len(split1)):
                        item1 = split1[i]
                        item2 = split2[i]
                        if item1 != item2:
                            dd[item1].append(item2)
    abbreviations_list = []
    for i in dd.keys():
        counter = Counter(dd[i])
        acronyms_counter = []
        for t in counter.keys():
            acronyms_counter.append(Acronym(t, counter[t]))
        acronyms_counter = [i for i in acronyms_counter if i.count > 1]
        if len(acronyms_counter) > 0:
            abbreviations_list.append(Abbreviation(i, acronyms_counter))
        # abbreviations_list.append(Abbreviation(abbreviation=i, standing_for=list(set(dd[i]))))

    return Abbreviations(abbreviations_list)


if __name__ == "__main__":
    csv_path = "/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/chant.csv"
    abbreviations = get_abbreviations(csv_path)
    with open('abbreviations.json', 'w', encoding='utf-8') as f:
        json.dump(abbreviations.to_dict(), f, ensure_ascii=False, indent=4)
