from typing import List

from dataclasses_json import dataclass_json

from database.database_dictionary import DatabaseDictionary, WordFrequencyDict, WordFrequency
from database.file_formats.importer.mondodi.simple_import import MonodiDocument
import json
import logging
import os
from collections import defaultdict

from dataclasses import dataclass

from database.file_formats.importer.mondodi.simple_import import simple_monodi_data_importer
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
from omr.text_matching.populate_db import MonodiImportStructure

logger = logging.getLogger(__name__)


def load_json(path):
    json_data = None

    with open(path, 'r') as fp:
        json_data = json.load(fp)
    return json_data


def populate(path):
    sentence = simple_monodi_data_importer(load_json(path))

    return sentence


def list_dirs(folder, dir=False):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))] if dir else os.listdir(folder)


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


def populate_look_up_dict(sentence, id, word_dict):
    for x in sentence:
        word_dict[x].append(id)


def check_word_based_similarity(sentence, word_dict):
    documents = []
    for x in sentence:
        if x in word_dict:
            documents += (set(word_dict[x]))
    from collections import Counter
    count = Counter(documents)
    return count





class WordDictionaryGenerator:
    def __init__(self, export_path):
        self.word_dict = defaultdict(list)
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        try:
            self.populate(export_path)
        except Exception as e:
            logger.error("Could not Load Document Database")
            logger.exception(e)

    def populate(self, path):
        for x in documents_gen(path):
            a = populate(x.data)
            populate_look_up_dict(a.get_text(self.text_normalizer).split(" "), x.document_id, self.word_dict)

    def get_sorted_frequency_word_list(self):
        freq_word_list = []
        for key, y in self.word_dict.items():
            freq_word_list.append(WordFrequency(word=key, frequency=len(y)))
        return WordFrequencyDict(freq_list=sorted(freq_word_list, key=lambda entry: entry.frequency, reverse=True))

    def write_to_json_file(self, path):
        freq_word_list: WordFrequencyDict = self.get_sorted_frequency_word_list()
        jso = DatabaseDictionary(dictionary=freq_word_list)
        with open(os.path.join(path, "default_dictionary.json"), 'w') as outfile:
            dump = json.dumps(jso.to_json(), indent=4)
            outfile.write(dump)

        # freq_word_list.
        pass


if __name__ == "__main__":
    a = WordDictionaryGenerator("/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/export")
    a.write_to_json_file(".")
