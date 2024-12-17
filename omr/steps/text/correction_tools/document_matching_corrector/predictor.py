import enum
import math
import os
import re
from dataclasses import dataclass

import edlib
import numpy as np

if __name__ == '__main__':
    import django


    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    from database.start_up.load_text_variants_in_memory import load_model
    load_model()
from database.start_up.load_text_variants_in_memory import lyrics, syllable_dictionary

from database.database_book_documents import DatabaseBookDocuments
from database.database_dictionary import DatabaseDictionary
from database.file_formats.book.document import Document
from database.file_formats.pcgts.page import Sentence
from database.model import Model, MetaId
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage, DatabaseBook
from typing import List, Optional, NamedTuple

from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.layout.predictor import PredictionResult
from omr.steps.step import Step
from omr.steps.text.hyphenation.hyphenator import Hyphenator, HyphenDicts, CombinedHyphenator
from omr.text_matching.populate_db import SimilarDocumentChecker
from tools.simple_gregorianik_text_export import Lyric_info
from database.file_formats.pcgts import Coords, PageScaleReference

from itertools import zip_longest



def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class ResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class SingleDocumentResult(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    matched_document: str
    page_id: str
    document_id: str
    document: Document
    book: DatabaseBook
    lyric_info: Lyric_info

    def to_dict(self):
        return {'similarText': self.matched_document,
                "page_id": self.page_id,
                "document_id": self.document_id}


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    documents: List[SingleDocumentResult]

    def to_dict(self):
        return {'docs': [s.to_dict() for s in self.documents]}

    def store_to_page(self):
        if len(self.documents) > 0:
            documents_loaded = DatabaseBookDocuments().load(self.documents[0].book)
        else:
            documents_loaded = None
        docs = documents_loaded.database_documents.documents if documents_loaded else None

        for doc in self.documents:
            lines = doc.document.get_page_line_of_document(book=doc.book)
            matched_lines = doc.matched_document.split("\n")
            for line_ind, line in enumerate(lines):
                line, page = line
                if line_ind < len(matched_lines):
                    #print("Now")
                    #print(line.text())
                    #print(matched_lines[line_ind])
                    line.sentence = Sentence.from_string(matched_lines[line_ind])
                    page.pcgts().page.annotations.connections.clear()
                    page.pcgts().to_file(page.file('pcgts').local_path())
            if documents_loaded:
                documents_loaded.database_documents.update_document_meta_infos(doc.lyric_info, doc.document.doc_id)
        if documents_loaded:
            documents_loaded.to_file(self.documents[0].book)
                # self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class sync_operation(enum.IntEnum):
    MATCH = enum.auto()
    MISMATCH = enum.auto()
    INSERT = enum.auto()
    DELETE = enum.auto()
    WHITESPACE = enum.auto()
    NEWLINE = enum.auto()


@dataclass
class synced_char:
    char_pred: str
    char_orig: str
    operation: sync_operation


@dataclass
class SyncedText:
    synced_text: List[synced_char]

    def get_current_line(self, ind):
        line = 0
        for ind2, i in enumerate(self.synced_text):
            if ind2 == ind:
                break
            if i.operation == i.operation.NEWLINE:
                line += 1
        return line

    def __iter__(self):
        return iter(self.synced_text)


def get_window(ind: int, synced_text: List[synced_char], min_match=1):
    left = []
    right = []
    ind_left = ind
    ind_right = ind
    o_item = synced_text[ind]
    l_match = 0
    r_match = 0
    while True:
        ind_right += 1
        if ind_right < len(synced_text):
            item = synced_text[ind_right]
            if item.operation == sync_operation.MATCH:
                right.append(item)
                r_match += 1

                if r_match >= min_match:
                    break

            elif item.operation == sync_operation.NEWLINE:
                break
            else:
                right.append(item)
        else:
            break

    while True:
        ind_left -= 1
        if ind_left >= 0:
            item = synced_text[ind_left]
            if item.operation == sync_operation.MATCH:
                left.insert(0, item)
                l_match += 1

                if l_match >= min_match:
                    break

            elif item.operation == sync_operation.NEWLINE:
                break
            else:
                left.insert(0, item)
        else:
            break
    return left, o_item, right


def get_word(ind: int, synced_text: List[synced_char]):
    indices_left = []
    indices_right = []

    left = []
    right = []
    ind_left = ind
    ind_right = ind
    o_item = synced_text[ind]
    while True:
        ind_right += 1
        if ind_right < len(synced_text):
            item = synced_text[ind_right]
            if item.operation == sync_operation.WHITESPACE:  # or item.operation == sync_operation.NEWLINE:
                break
            else:
                right.append(item)
                indices_right.append(ind_right)
        else:
            break

    while True:
        ind_left -= 1
        if ind_left >= 0:
            item = synced_text[ind_left]
            if item.operation == sync_operation.WHITESPACE:  # or item.operation == sync_operation.NEWLINE:
                break
            else:
                left.insert(0, item)
                indices_left.insert(0, ind_left)

        else:
            break

    return left + [o_item] + right, indices_left + [ind] + indices_right


def best_new_line_position(word: List[synced_char], word_indices: List[int], hyphen: Hyphenator,
                           synced_list: SyncedText, current_new_line_ind):
    total_text = ""
    # print("wtf")
    for i in word:
        total_text += i.char_orig
    #    print(i.operation)
    #    print(i.char_orig)
    str_word = "".join([x.char_orig for x in word])
    syllables = hyphen.apply_to_word(str_word)
    indexes = [i for i, x in enumerate(syllables) if x == "-"]
    for ind, i in enumerate(indexes):
        indexes[ind] = i - ind
    global_indices = [word_indices[0]] + [word_indices[i] for i in indexes] + [word_indices[-1] + 1]
    scoring = []
    for ind, i in enumerate(global_indices):
        if ind + 1 >= len(global_indices):
            break
        next_item = global_indices[ind + 1]
        sub_l = synced_list.synced_text[i: next_item]
        syl = ''
        score = 0
        for t in sub_l:
            if t.operation == t.operation.MATCH:
                score += 1
            syl += t.char_orig
        scoring.append(score)

    current_new_line_ind = [i for i in global_indices if current_new_line_ind - i > 0][-1]
    index_of = global_indices.index(current_new_line_ind)
    score_left = [(i, index_of - 1 + ind_1) for ind_1, i in enumerate(synced_list.synced_text[index_of - 1: index_of])
                  if
                  i.operation == i.operation.MATCH]
    score_right = [(i, index_of + ind_1) for ind_1, i in enumerate(synced_list.synced_text[index_of: index_of + 1]) if
                   i.operation == i.operation.MATCH]
    loc_left = [synced_list.get_current_line(ind) for i, ind in score_left]
    loc_right = [synced_list.get_current_line(ind) for i, ind in score_right]

    for i in [(index_of - 1, index_of), (index_of, index_of + 1)]:
        pass

    return indexes, global_indices


def best_new_line_position2(word: List[synced_char], word_indices: List[int], hyphen: Hyphenator,
                            synced_list: SyncedText, current_new_line_ind):
    total_text = ""
    # print("wtf")
    for i in word:
        total_text += i.char_orig
    #    print(i.operation)
    #    print(i.char_orig)
    # print(total_text)
    str_word = "".join([x.char_orig for x in word])
    syllables = hyphen.apply_to_word(str_word)
    # print(syllables)
    indexes = [i for i, x in enumerate(syllables) if x == "-"]
    for ind, i in enumerate(indexes):
        indexes[ind] = i - ind
    # print("indexes")

    # print(indexes)
    global_indices = [word_indices[0]] + [word_indices[i] for i in indexes] + [word_indices[-1] + 1]
    str2 = ""
    for i in synced_list.synced_text[global_indices[0]: global_indices[-1]]:
        str2 += i.char_orig if i.char_orig else " "
    #print(f"str2 {str2}")
    #print(f"global_indices {global_indices}")

    #previous_line = \
    #    [synced_list.get_current_line(ind) for ind, i in enumerate(synced_list.synced_text[:current_new_line_ind])][-1]
    previous_lines = \
        [synced_list.get_current_line(ind) for ind, i in enumerate(synced_list.synced_text[:current_new_line_ind])]

    #print(f"previous_line {previous_line}")
    current_new_line_ind2 = [i for i in global_indices if current_new_line_ind - i >= 0][-1]

    index_of = global_indices.index(current_new_line_ind2)
    second = global_indices[index_of + 1]
    #print(f"current_new_line_ind2 {current_new_line_ind2} second {second}")
    loc = [synced_list.get_current_line(ind) for ind in range(current_new_line_ind2, second) if
           synced_list.synced_text[ind].operation == sync_operation.MATCH]
    previous_line = previous_lines[-1] if len(previous_lines) > 0 else loc[0] if len(loc) > 0 else None
    #print(f"locations {loc}")
    # if syllables == "spe-ra-vi":
    #    print(loc)
    #    print("spe-ra-vispe-ra-vi")
    if previous_line:
        if len(loc) > 0:
            if len(set(loc)) == 1:
                if loc[0] == previous_line:
                    return second
                else:
                    return current_new_line_ind2
            else:
                return current_new_line_ind2
        else:
            return second
    return current_new_line_ind2

def align_documents2(b: Lyric_info, document: Document, doc_text: str, doc_list: List[str],
                    normalizer: LyricsNormalizationProcessor, hyphen: Hyphenator):
    from collections import Counter

    def most_common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    text_pred = [(t, ind) for ind, i in enumerate(doc_list) for t in i]
    #print(text_pred)
    # remove whitespaces
    text_pred = [(i, t) for i, t in text_pred if i != ' ' and i != '-']

    #print(text_pred)
    only_text = "".join([i[0] for i in text_pred])
    text_gt = " ".join(b.latine.split(" "))
    #text_gt = hyphen.apply_to_sentence(text_gt)
    text_gt = text_gt.replace("-", " ")
    #print(text_gt)
    text_gt = [(t, ind) for ind, i in enumerate(text_gt.split(" ")) for t in i]
    #text_words = [(i, ind) for ind, i in enumerate(text_gt.split(" "))]

    only_text_gt = "".join([i[0] for i in text_gt])
    #print(only_text_gt)
    #print(only_text_gt)
    #print(only_text)
    ed = edlib.align(only_text.lower(), only_text_gt.lower(), mode="SHW", task="path")
    print("___")
    print(only_text_gt)
    print(only_text)
    print("___")
    gap_char = "#"
    nice = edlib.getNiceAlignment(ed, only_text, only_text_gt, gap_char)
    @dataclass
    class SyncedCharInfo:
        char_pred: str
        char_gt: str
        operation: sync_operation
        pred_line: int
        gt_char_of_word: int
        #gt_syllable_after: bool

    for ind, i in enumerate(nice["query_aligned"]):
        if i == gap_char:
            text_pred.insert(ind, (gap_char, None))

    for ind, i in enumerate(nice["target_aligned"]):
        if i == gap_char:
            text_gt.insert(ind, (gap_char, None))
    synced_text = []
    #print(nice["query_aligned"])
    #print(nice["matched_aligned"])
    #print(nice["target_aligned"])
    #print(text_pred)
    #print(text_gt)

    for i in zip(text_pred, nice["matched_aligned"], text_gt):
        p_c = i[0]
        op_c = i[1]
        o_c = i[2]
        if op_c == "|":
            synced_text.append(SyncedCharInfo(char_pred=p_c[0], char_gt=o_c[0], operation=sync_operation.MATCH, pred_line=p_c[1], gt_char_of_word=o_c[1]))
        elif op_c == ".":
            synced_text.append(SyncedCharInfo(char_pred=p_c[0], char_gt=o_c[0], operation=sync_operation.MISMATCH, pred_line=p_c[1], gt_char_of_word=o_c[1]))
        elif op_c == gap_char:
            if p_c[0] == gap_char:
                synced_text.append(SyncedCharInfo(char_pred=None, char_gt=o_c[0], operation=sync_operation.INSERT,
                                                  pred_line=None, gt_char_of_word=o_c[1]))
            elif o_c[0] == gap_char:
                synced_text.append(SyncedCharInfo(char_pred=p_c[0], char_gt=None, operation=sync_operation.DELETE,
                                                  pred_line=p_c[1], gt_char_of_word=None))

    #print(synced_text)
    word_list=[]
    word = []
    prev_word = 0
    for i in synced_text:
        if i.gt_char_of_word != prev_word:
            if i.gt_char_of_word is not None:
                #print("")
                prev_word = i.gt_char_of_word
                word_list.append(word)
                word = []
        word.append(i)
    word_list.append(word)
        #print(f'CGT: {i.char_gt}, pwt: {i.char_pred}, line: {i.pred_line} word: {i.gt_char_of_word}')
    lines = []
    c_line =[]
    c_line_index = 0
    #print(word_list)
    for w in word_list:
        w = [i for i in w if i.char_gt is not None]
        word =  "".join([i.char_gt for i in w if i.char_gt is not None])
        hyphen_word = hyphen.apply_to_word(word)
        syl_gap_char_indexes = [ind for ind, i in enumerate(hyphen_word) if i == "-"]
        syl_gap_char_indexes = [i - ind for ind, i in enumerate(syl_gap_char_indexes)]
        syl_gap_char_indexes.append(len(word))
        prev = 0
        #print(word)
        #print(hyphen_word)
        for i in syl_gap_char_indexes:
            syl = w[prev:i]
            #print("123")
            #print(syl)
            syl_line = [s_c.pred_line for s_c in syl if s_c.pred_line is not None]
            if len(syl_line) > 0:
                line_index = most_common(syl_line)
            else:
                line_index = c_line_index
            prev = i
            if line_index == c_line_index:
                c_line.append(syl)
            else:
                c_line_index = line_index
                lines.append(c_line)
                c_line = []
                c_line.append(syl)

        c_line.append([SyncedCharInfo(char_gt=" ", char_pred=" ", operation=None, pred_line=None, gt_char_of_word=None)])
    lines.append(c_line)
    doc = ""
    for t in lines:
        l = "".join([c.char_gt for i in t for c in i])
        doc += l.lstrip().rstrip()
        doc += "\n"
    ##print(b.latine)
    ##print(doc)

    aligned_text_elements = []
    single_document_result = []

    #for i in doc.split("\n"):
    #    i = hyphen.apply_to_sentence(i)
    #    aligned_text_elements.append(i)
    #aligned_text = "\n".join(aligned_text_elements)

    return doc


def align_documents(b: Lyric_info, document: Document, doc_text: str, doc_list: List[str],
                    normalizer: LyricsNormalizationProcessor, hyphen: Hyphenator):
    doc_text = doc_text
    orig_text = b.latine
    doc_text = normalizer.apply(doc_text).replace(" ", "")
    orig_text = normalizer.apply(orig_text)
    orig_text_wo_ws = orig_text.replace(" ", "")

    ed = edlib.align(doc_text, orig_text_wo_ws, mode="SHW", task="path")
    locations = ed["locations"][0]
    gap_char = "#"
    nice = edlib.getNiceAlignment(ed, doc_text, orig_text_wo_ws, gap_char)
    synced_text = []
    # print("\n".join(nice.values()))
    ##print(len(nice["query_aligned"]))
    # print(len(nice["matched_aligned"]))
    # print(len(nice["target_aligned"]))

    for i in zip(nice["query_aligned"], nice["matched_aligned"], nice["target_aligned"]):
        p_c = i[0]
        op_c = i[1]
        o_c = i[2]
        if op_c == "|":
            synced_text.append(synced_char(char_pred=p_c, char_orig=o_c, operation=sync_operation.MATCH))
        elif op_c == ".":
            synced_text.append(synced_char(char_pred=p_c, char_orig=o_c, operation=sync_operation.MISMATCH))
        elif op_c == gap_char:
            if p_c == gap_char:
                synced_text.append(synced_char(char_pred="", char_orig=o_c, operation=sync_operation.INSERT))
            elif o_c == gap_char:
                synced_text.append(synced_char(char_pred=p_c, char_orig="", operation=sync_operation.DELETE))
    # sync whitespaces
    pointer = 0

    # print("___2")
    # print(synced_text[-1].char_orig)
    # print(synced_text[-1].char_pred)
    # print("___2")

    # print(len(synced_text))
    # print(orig_text)
    # print("___")

    for i in " ".join(orig_text.split()):
        if pointer < len(synced_text):
            # print(len(synced_text))
            ##print(pointer)
            # print(synced_text[pointer].operation)

            # print(i)
            # print(synced_text[pointer].char_orig)

            while pointer < len(synced_text) and synced_text[pointer].operation == sync_operation.DELETE:
                pointer += 1

            if i == " ":
                synced_text.insert(pointer,
                                   synced_char(char_pred="", char_orig="", operation=sync_operation.WHITESPACE))
            pointer += 1

    # sync newlines
    new_line_string = ""
    newline_token = "€"
    for i in doc_list:
        new_line_string += normalizer.apply(i).replace(" ", "")
        new_line_string += newline_token  # newline
    # print(new_line_string)
    pointer = 0
    for i in new_line_string:
        if pointer < len(synced_text):
            while pointer < len(synced_text) and (synced_text[pointer].operation == sync_operation.WHITESPACE or synced_text[
                pointer].operation == sync_operation.INSERT):
                pointer += 1
            if i == newline_token:
                synced_text.insert(pointer, synced_char(char_pred="", char_orig="", operation=sync_operation.NEWLINE))

            pointer += 1

    word = ""

    def generate_linestring(synced_text):
        text_combined = ""
        for i in synced_text:
            # print(f'o: {i.char_orig} p: {i.char_pred}')
            ##print(i.operation)
            if i.operation == sync_operation.WHITESPACE:
                text_combined += " "
            elif i.operation == sync_operation.NEWLINE:
                text_combined += "€"
            else:
                text_combined += i.char_orig
        return text_combined

    o_string = generate_linestring(synced_text)
    max_moves = 50
    c_move = 0
    while True:
        text_combined = generate_linestring(synced_text)
        #print(text_combined)
        #print(o_string)
        change = False
        prev = None

        for ind, i in enumerate(synced_text):
            next_item = synced_text[ind + 1] if len(synced_text) > ind + 1 else None
            if i.operation == sync_operation.NEWLINE:
                if (prev != None and prev.operation == sync_operation.WHITESPACE) or (
                        next_item != None and next_item.operation == sync_operation.WHITESPACE):
                    continue
                else:
                    #print("")
                    #print("")
                    word, word_indices = get_word(ind, synced_text)
                    new_line_ind = best_new_line_position2(word, word_indices, hyphen, SyncedText(synced_text), ind)
                    word_str = [s.char_orig if s.char_orig else s.operation.name for s in word]
                    #print("Next Lb")
                    #print(f"LenSy {len(synced_text)}, ind: {ind}, newlineind, {new_line_ind} word {word_str}")

                    if new_line_ind != ind:
                        if new_line_ind > ind:
                            text3 = ""

                            for g in synced_text:
                                text3 += g.char_orig if g.char_orig else g.operation.name
                            #print(text3)

                            ##print(synced_text[new_line_ind - 1])
                            # print(synced_text[new_line_ind])
                            synced_text.insert(new_line_ind, synced_char("", "", sync_operation.NEWLINE))

                            # print(synced_text[new_line_ind +1])

                            del synced_text[ind]
                            # print("After:")
                            v = ""

                            for t in synced_text:
                                v += str(t.operation.value)
                            #print(v)
                            #print(
                            #    f"LenSAy {len(synced_text)}, ind: {ind}, newlineind, {new_line_ind}  word {word_str} ")

                        else:
                            # print("12345678sy")
                            # v = ""
                            # for t in synced_text:
                            #    v += str(t.operation.value)
                            # print(v)

                            synced_text.insert(new_line_ind, synced_char("", "", sync_operation.NEWLINE))
                            del synced_text[ind + 1]
                            v = ""
                            for t in synced_text:
                                v += str(t.operation.value)
                            #print(v)
                            #print(f"LenSpy {len(synced_text)}, ind: {ind}, newlineind, {new_line_ind}  word {word_str}")
                        if text_combined != generate_linestring(synced_text):
                            if max_moves > c_move:
                                change = True
                            c_move += 1
                            break

            prev = i

        if not change:
            break
    aligned_string = generate_linestring(synced_text)
    #print(doc_text)
    #print(new_line_string)
    #print(o_string)
    #print(aligned_string)

    aligned_text = ""

    for i in synced_text:
        if i.operation == i.operation.WHITESPACE:
            aligned_text += " "
        elif i.operation == i.operation.NEWLINE:
            aligned_text += "\n"
        else:
            aligned_text += i.char_orig
    #print(aligned_text)

    return aligned_text

    #    prev = i

    """
    if len(i) - locations[1] <= max(5, math.ceil(0.1 * len(i))):
        ed = edlib.align(doc_text, orig_text, mode="NW", task="path")
    else:
        def get_nearest_higher_number(ind: int, w_array: List[int]):
            for i in w_array:
                if i > ind:
                    return i
            return ind

        locations = ed["locations"][0]
        end_location = get_nearest_higher_number(locations[1], whitespaces)
        orig_text = orig_text[locations[0]:end_location + 1]
        ed = edlib.align(doc_text, orig_text, mode="NW", task="path")
    print(edlib.getNiceAlignment()
    pass
    """


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        from omr.steps.text.correction_tools.document_matching_corrector.meta import Meta

        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        # self.document_similar_tester = SimilarDocumentChecker()
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        self.database_hyphen_dictionary = syllable_dictionary

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None, gt_book: DatabaseBook = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        single_document_result = []
        all_docs: List[Document] = []
        # if self.database_hyphen_dictionary is None:
        #    db = DatabaseDictionary.load(book=book)
        #    self.database_hyphen_dictionary = db.to_hyphen_dict()
        hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                            right=1)

        if self.document_id is not None:
            document: Document = documents.database_documents.get_document_by_id(self.document_id)
            all_docs.append(document)
        else:
            for i in pages:
                all_docs += documents.database_documents.get_documents_of_page(i.pcgts().page)
        for ind_doc, doc in enumerate(all_docs):
            document = doc
            text = document.get_text_of_document(book)
            text_list = document.get_text_list_of_line_document(book)

            text = self.text_normalizer.apply(text)
            lowest_ed = 999999
            lowest_text = ""
            # ed = edlib.align("abc", "abc", mode="SHW", k=lowest_ed)
            lyric_info: Lyric_info = None
            if gt_book:
                documents2 = DatabaseBookDocuments().load(gt_book)
                document2 = documents2.database_documents.get_document_by_b_uid(document.get_book_u_id())
                print(document.get_book_u_id())
                print(text)
                #text_list2 = document2.get_text_list_of_line_document(gt_book)
                text2 = document2.get_text_of_document(gt_book)
                text2 = self.text_normalizer.apply(text2)
                lyric_info = Lyric_info(latine=text2, id="1", index="1", meta_info="1", meta_infos_extended = None, variants = None)
                print(text)
                print(text2)
            else:
                for b in lyrics.lyrics:
                    b: Lyric_info = b
                    text2 = self.text_normalizer.apply(b.latine)
                    ed = edlib.align(text.replace(" ", ""), text2.replace(" ", ""), mode="SHW", k=lowest_ed)
                    if 0 < ed["editDistance"] < lowest_ed:
                        lowest_ed = ed["editDistance"]
                        lowest_text = text2
                        lyric_info = b
                    elif text.replace(" ", "") == text2.replace(" ", ""):
                        lowest_ed = 0
                        lowest_text = text2
                        lyric_info = b

            if lyric_info:
                #print(lyric_info.to_json())
                try:
                    aligned_text = align_documents2(lyric_info, document, text, text_list, self.text_normalizer, hyphen)
                except Exception as e:
                    aligned_text = ""
                    for i in text_list:
                        aligned_text += i + "\n"
                #aligned_text = align_documents(lyric_info, document, text, text_list, self.text_normalizer, hyphen)

            else:
                aligned_text = ""
                for i in text_list:
                    aligned_text += i + "\n"
            """
            text = text.replace(" ", "")
            for i in [lowest_text]:
                orig_text = i
                if len(i) == 0:
                    single_document_result.append(SingleDocumentResult(matched_document="",
                                                                       page_id=document.start.page_id,
                                                                       document_id=document.doc_id,
                                                                       document=document,
                                                                       book=book))
                    continue
                i = " ".join(i.split())
                orig = i
                whitespaces = [ind for ind, i in enumerate(i) if i == " "]
                i = i.replace(" ", "")
                ed = edlib.align(text, i, mode="SHW", task="path")
                locations = ed["locations"][0]
                if len(i) - locations[1] <= max(5, math.ceil(0.1 * len(i))):
                    ed = edlib.align(text, i, mode="NW", task="path")
                else:
                    def get_nearest_higher_number(ind: int, w_array: List[int]):
                        for i in w_array:
                            if i > ind:
                                return i
                        return ind

                    locations = ed["locations"][0]
                    end_location = get_nearest_higher_number(locations[1], whitespaces)
                    i = i[locations[0]:end_location + 1]
                    ed = edlib.align(text, i, mode="NW", task="path")
                cigar = ed["cigar"]
                if i != "":
                    try:
                        a = edlib.getNiceAlignment(ed, text, i)
                    except Exception as e:
                        raise e
                    line_index = 0
                    aligned_text = ""
                    aligned_ocr_text = ""
                    for aligned in range(len(a["matched_aligned"])):
                        if len(whitespaces) > 0:
                            if whitespaces[0] <= len(aligned_text.replace("\n", "")):
                                aligned_text += " "
                                del (whitespaces[0])
                        aligned_text += a['target_aligned'][aligned] if a['target_aligned'][aligned] != "-" else ""
                        aligned_ocr_text += a['query_aligned'][aligned] if a['query_aligned'][aligned] != "-" else ""

                        if len(text_list) > line_index:
                            if text_list[line_index].replace(" ", "").replace("-", "") == aligned_ocr_text:
                                aligned_ocr_text = ""
                                aligned_text += "\n"
                                line_index += 1
                        else:
                            aligned_text = aligned_text[:-1].rstrip("\n") + aligned_text[-1:] + "\n"

                    def correct_line_splits(aligned_text: str, orig_text: str):
                        orig_word = orig_text.split(" ")
                        lines = aligned_text.split("\n")
                        lines = [i.strip() for i in lines if len(i.strip()) > 0]
                        stop_while = False
                        while not stop_while:
                            for line_ind in range(len(lines)):
                                line = lines[line_ind]
                                words_l1 = line.lstrip(" ").rstrip(" ").split(" ")
                                last_word = words_l1[-1]
                                if line_ind + 1 < len(lines) - 1:
                                    words_l2 = lines[line_ind + 1].lstrip(" ").rstrip(" ").split(" ")
                                    if last_word + words_l2[0] in orig_word:
                                        hyphenated_word = hyphen.apply_to_word(last_word + words_l2[0]).split("-")
                                        if words_l2[0] == hyphenated_word[-1] or words_l2[0] == '':
                                            pass
                                        else:
                                            lines[line_ind] = lines[line_ind][:-len(last_word)] + last_word + words_l2[
                                                0]
                                            lines[line_ind + 1] = lines[line_ind + 1][
                                                                  len(words_l2[0]):]  # .replace(words_l2[0], "")
                                            break
                                else:
                                    stop_while = True
                                    break
                        lines = [i.strip() for i in lines if len(i) > 0]
                        a_text = "\n".join(lines)
                        return a_text

                    aligned_text = correct_line_splits(aligned_text, orig_text)
                    """
            aligned_text_elements = []

            for i in aligned_text.split("\n"):
                i = hyphen.apply_to_sentence(i)
                aligned_text_elements.append(i)
            aligned_text = "\n".join(aligned_text_elements)

            single_document_result.append(SingleDocumentResult(matched_document=aligned_text,
                                                               page_id=document.start.page_id,
                                                               document_id=document.doc_id,
                                                               document=document,
                                                               book=book,
                                                               lyric_info=lyric_info))  # .append(aligned_text)

            percentage = (ind_doc) / len(all_docs)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=ind_doc + 1, n_pages=len(all_docs))

        yield Result(single_document_result)


if __name__ == '__main__':
    from omr.steps.step import Step, AlgorithmTypes
    from ommr4all.settings import BASE_DIR
    import random
    import cv2
    import matplotlib.pyplot as plt
    from shared.pcgtscanvas import PcGtsCanvas
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    random.seed(1)
    np.random.seed(1)

    book = DatabaseBook('mul_2_end_w_gt_symbols_w_finetune_no_pp_gt_text_seg3')
    book2 = DatabaseBook('mul_2_rsync_gt')

    meta = Step.meta(AlgorithmTypes.TEXT_DOCUMENT)
    #load = "i/french14/text_guppy/text_guppy")
    from database.model import Model

    #model = Model.from_id_str("e/mul_2_rsync_gt/text_guppy/2023-07-05T16:32:45")
    model = meta.default_model_for_book(book)
    # model = Model(
    #    MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical',
    ##                            meta.type()))
    pred = Predictor(AlgorithmPredictorSettings(model))
    ps: List[PredictionResult] = list(pred.predict(book.pages(), gt_book=book2))
    for i in ps:
        i.store_to_page()
        print("123")
        print(i)


