from typing import List

from dataclasses_json import dataclass_json

from database.database_dictionary import DatabaseDictionary, WordFrequencyDict, WordFrequency
from database.file_formats.importer.mondodi.simple_import import MonodiDocument
import json
import logging
import os
from collections import defaultdict, Counter

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


def populate_look_up_dict(sentence, word_dict, t1, t2):
    for x in sentence:
        word = t2.apply(x)
        if "\n" in word:
            continue
        word_dict[word].append(t1.apply(x))


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
        self.text = ""
        self.bigram_word_dict = defaultdict(list)
        self.sentence_list = defaultdict(lambda: 0)
        self.text_normalizer1 = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.SYLLABLES))
        self.text_normalizer2 = LyricsNormalizationProcessor(
            LyricsNormalizationParams(LyricsNormalization.WORDS,
                                      True,
                                      False,
                                      True,
                                      True,
                                      ))

        try:
            self.populate(export_path)
        except Exception as e:
            logger.error("Could not Load Document Database")
            logger.exception(e)

    def populate(self, path):
        for x in documents_gen(path):
            if os.path.exists(x.data):
                a = populate(x.data)
                self.text += a.get_text() + " "
                for x in a.get_text().split("\n"):
                    x = x.strip().lower().replace("-", "")
                    if len(x) > 0:
                        self.sentence_list[x] = self.sentence_list[x] + 1
                populate_look_up_dict(a.get_text().split(" "), self.word_dict, self.text_normalizer1, self.text_normalizer2)

    def populate_bigram_look_up_dict(self, sentence, word_dict, t1, t2):
        last_word = None
        last_word_hyp = None
        for x in sentence:
            word = t2.apply(x)
            if "\n" in word:
                continue
            if last_word is not None:
                word_dict[last_word + " " + word].append(last_word_hyp + " " + t1.apply(x))

            last_word = word
            last_word_hyp = t1.apply(x)

    def populate_bigram(self, path):
        for x in documents_gen(path):
            a = populate(x.data)
            self.populate_bigram_look_up_dict(a.get_text().split(" "), self.bigram_word_dict,
                                              self.text_normalizer1, self.text_normalizer2)
        # print(self.get_sorted_frequency_bigram_word_list())

    def get_sorted_frequency_bigram_word_list(self):
        freq_word_list = []
        for key, y in self.bigram_word_dict.items():
            c = Counter(y)
            freq_word_list.append(WordFrequency(word=key, frequency=len(y), hyphenated=c.most_common(1)[0][0]))
        return WordFrequencyDict(freq_list=sorted(freq_word_list, key=lambda entry: entry.frequency, reverse=True))

    def get_sorted_frequency_word_list(self):
        freq_word_list = []
        for key, y in self.word_dict.items():
            c = Counter(y)
            freq_word_list.append(WordFrequency(word=key, frequency=len(y), hyphenated=c.most_common(1)[0][0]))
        return WordFrequencyDict(freq_list=sorted(freq_word_list, key=lambda entry: entry.frequency, reverse=True))

    def write_to_json_file(self, path):
        freq_word_list: WordFrequencyDict = self.get_sorted_frequency_word_list()
        jso = DatabaseDictionary(dictionary=freq_word_list)
        with open(os.path.join(path, "default_dictionary.json"), 'w') as outfile:
            dump = json.dumps(jso.to_json(), indent=4)
            outfile.write(dump)

        # freq_word_list.
        pass

    def write_bigram_to_file(self, path):
        freq_word_list: WordFrequencyDict = self.get_sorted_frequency_bigram_word_list()
        with open(os.path.join(path, "bigram_default_dictionary.json"), 'w') as outfile:
            for x in freq_word_list.freq_list:
                outfile.write(x.word + " " + str(x.frequency) + "\n")

    def get_language_model_text(self):
        return self.text

    def write_language_model_text(self, path):
        text = self.text
        text = self.text_normalizer2.apply(text=text)
        text = text.replace("\n", "")
        with open(os.path.join(path, "text_language_model.json"), 'w') as outfile:
            outfile.write(text)

    def write_sentence_file(self, path, seperator="$"):
        freq_word_list: WordFrequencyDict = self.get_sorted_frequency_bigram_word_list()
        with open(os.path.join(path, "sentence_dictionary.json"), 'w') as outfile:
            for x in self.sentence_list.keys():
                outfile.write(x + seperator + str(self.sentence_list[x]) + "\n")
if __name__ == "__main__":
    a = WordDictionaryGenerator("/home/alexanderh/Downloads/mc_export/export/")
    a.write_to_json_file(".")
    a.populate_bigram("/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/export")
    a.write_bigram_to_file(".")
    #a.write_language_model_text(".")
    a.write_sentence_file(".")
