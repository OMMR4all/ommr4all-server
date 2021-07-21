import json
import os
from collections import defaultdict

from dataclasses import dataclass

from database.file_formats.importer.mondodi.simple_import import simple_monodi_data_importer
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization


def load_json(path):
    json_data = None

    with open(path, 'r') as fp:
        json_data = json.load(fp)
    return json_data


def populate(path):
    sentence = simple_monodi_data_importer(load_json(path))

    return sentence


@dataclass
class MonodiImportStructure:
    source_meta: str
    data: str
    document_meta: str
    document_id: str


def list_dirs(folder, dir=False):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))] if dir else os.listdir(folder)


def documents_gen():
    from ommr4all.settings import BASE_DIR

    b_dir = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'monodi_db', 'export')
    dir = list_dirs(b_dir, True)
    for x in dir:
        s_dir = os.path.join(b_dir, x)
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


class SimilarDocumentChecker:
    def __init__(self):
        self.word_dict = defaultdict(list)
        self.document_dict = {}
        self.document_path = {}
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        self.populate()

    def populate(self):
        for x in documents_gen():
            a = populate(x.data)
            populate_look_up_dict(a.get_text(self.text_normalizer).split(" "), x.document_id, self.word_dict)
            self.document_dict[x.document_id] = a
            self.document_path[x.document_id] = x

    def check_word_based_similarity(self, sentence):
        documents = []
        for x in sentence:
            if x in self.word_dict:
                documents += (set(self.word_dict[x]))
        from collections import Counter
        count = Counter(documents)
        return count


if __name__ == "__main__":
    word_dict = defaultdict(list)
    document_dict = {}
    document_path = {}
    document_meta = {}
    text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))

    for x in documents_gen():
        b = load_json(x.document_meta)
        # if b["dokumenten_id"] == "Pa 1235-253-1":
        #
        #    print("123")
        a = populate(x.data)

        document_meta[x.document_id] = b["dokumenten_id"]
        text = a.get_text(text_normalizer).split(" ")

        # if b["dokumenten_id"] == "Pa 1235-253-1":
        #    print(b)
        #    print(text)

        populate_look_up_dict(text, x.document_id, word_dict)
        document_dict[x.document_id] = a
        document_path[x.document_id] = x

        # print(x)
        #    douo-ieo-si-seoein-ceor-ta-pe-ho-rib

    counter = check_word_based_similarity(["lux", "aduenit", "ueneranda", "lux", "in", "chrois"], word_dict)
    for key, count in counter.most_common(10):
        print(key)
        print(count)
        print(document_path[key].data)
        print(document_dict[key].sentence)
        print(document_dict[key].get_text(text_normalizer).split(" "))
    pass
