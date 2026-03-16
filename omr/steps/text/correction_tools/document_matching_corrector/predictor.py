import enum
import os
from dataclasses import dataclass

import edlib
import numpy as np
from loguru import logger

if __name__ == '__main__':
    import django


    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from database.start_up.load_text_variants_in_memory import lyrics_store

from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.book.document import Document
from database.file_formats.pcgts.page import Sentence
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage, DatabaseBook
from typing import List, Optional
from omr.steps.layout.predictor import PredictionResult
from omr.steps.text.hyphenation.hyphenator import Hyphenator, HyphenDicts, CombinedHyphenator
from tools.simple_gregorianik_text_export import Lyric_info

from itertools import zip_longest



def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@dataclass(frozen=True)
class SingleDocumentResult(AlgorithmPredictionResult):
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
    def store_to_page(self):
        pass

@dataclass(frozen=True)
class Result(AlgorithmPredictionResult):
    documents: List[SingleDocumentResult]

    def to_dict(self):
        return {'docs': [s.to_dict() for s in self.documents]}

    def store_to_page(self):
        logger.info("Saving to page")
        if len(self.documents) > 0:
            documents_loaded = DatabaseBookDocuments().load(self.documents[0].book)
        else:
            documents_loaded = None

        for doc in self.documents:
            lines = doc.document.get_page_line_of_document(book=doc.book)
            matched_lines = doc.matched_document.split("\n")
            for line_ind, line in enumerate(lines):
                line, page = line
                if line_ind < len(matched_lines):

                    line.sentence = Sentence.from_string(matched_lines[line_ind])
                    page.pcgts().page.annotations.connections.clear()
                    page.pcgts().to_file(page.file('pcgts').local_path())
            if documents_loaded:
                documents_loaded.database_documents.update_document_meta_infos(doc.lyric_info, doc.document.doc_id)
        if documents_loaded:
            documents_loaded.to_file(self.documents[0].book)


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
            if item.operation == sync_operation.WHITESPACE:
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
            if item.operation == sync_operation.WHITESPACE:
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
    for i in word:
        total_text += i.char_orig
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

    return indexes, global_indices


def best_new_line_position2(word: List[synced_char], word_indices: List[int], hyphen: Hyphenator,
                            synced_list: SyncedText, current_new_line_ind):
    total_text = ""
    for i in word:
        total_text += i.char_orig
    str_word = "".join([x.char_orig for x in word])
    syllables = hyphen.apply_to_word(str_word)
    indexes = [i for i, x in enumerate(syllables) if x == "-"]
    for ind, i in enumerate(indexes):
        indexes[ind] = i - ind
    global_indices = [word_indices[0]] + [word_indices[i] for i in indexes] + [word_indices[-1] + 1]
    str2 = ""
    for i in synced_list.synced_text[global_indices[0]: global_indices[-1]]:
        str2 += i.char_orig if i.char_orig else " "

    previous_lines = \
        [synced_list.get_current_line(ind) for ind, i in enumerate(synced_list.synced_text[:current_new_line_ind])]

    current_new_line_ind2 = [i for i in global_indices if current_new_line_ind - i >= 0][-1]

    index_of = global_indices.index(current_new_line_ind2)
    second = global_indices[index_of + 1]
    loc = [synced_list.get_current_line(ind) for ind in range(current_new_line_ind2, second) if
           synced_list.synced_text[ind].operation == sync_operation.MATCH]
    previous_line = previous_lines[-1] if len(previous_lines) > 0 else loc[0] if len(loc) > 0 else None
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
    text_pred = [(i, t) for i, t in text_pred if i.strip() != '' and i != '-']

    only_text = "".join([i[0] for i in text_pred])
    text_gt = " ".join(b.latine.split(" "))
    text_gt = text_gt.replace("-", " ")
    text_gt = [(t, ind) for ind, i in enumerate(text_gt.split(" ")) for t in i]

    only_text_gt = "".join([i[0] for i in text_gt])

    ed = edlib.align(only_text.lower(), only_text_gt.lower(), mode="HW", task="path")

    start_idx, end_idx = ed["locations"][0]
    text_gt_cropped = text_gt[start_idx: end_idx + 1]
    only_text_gt_cropped = only_text_gt[start_idx: end_idx + 1]
    gap_char = "#"

    ed_final = edlib.align(only_text.lower(), only_text_gt_cropped.lower(), mode="NW", task="path")
    nice = edlib.getNiceAlignment(ed_final, only_text, only_text_gt_cropped, gap_char)
    print("___")
    print(f"Original: {only_text_gt}")
    print(f"Fragment: {only_text_gt_cropped}")
    print(f"To Map: {only_text}")
    print("___")
    @dataclass
    class SyncedCharInfo:
        char_pred: str
        char_gt: str
        operation: sync_operation
        pred_line: int
        gt_char_of_word: int

    for ind, i in enumerate(nice["query_aligned"]):
        if i == gap_char:
            text_pred.insert(ind, (gap_char, None))

    for ind, i in enumerate(nice["target_aligned"]):
        if i == gap_char:
            text_gt.insert(ind, (gap_char, None))
    synced_text = []

    for i in zip(text_pred, nice["matched_aligned"], text_gt_cropped):
        p_c = i[0]
        op_c = i[1]
        o_c = i[2]
        if op_c == "|":
            synced_text.append(
                SyncedCharInfo(char_pred=p_c[0], char_gt=o_c[0], operation=sync_operation.MATCH, pred_line=p_c[1],
                               gt_char_of_word=o_c[1]))
        elif op_c == ".":
            synced_text.append(
                SyncedCharInfo(char_pred=p_c[0], char_gt=o_c[0], operation=sync_operation.MISMATCH, pred_line=p_c[1],
                               gt_char_of_word=o_c[1]))
        elif op_c == gap_char:
            if p_c[0] == gap_char:
                synced_text.append(SyncedCharInfo(char_pred=None, char_gt=o_c[0], operation=sync_operation.INSERT,
                                                  pred_line=None, gt_char_of_word=o_c[1]))
            elif o_c[0] == gap_char:
                synced_text.append(SyncedCharInfo(char_pred=p_c[0], char_gt=None, operation=sync_operation.DELETE,
                                                  pred_line=p_c[1], gt_char_of_word=None))

    word_list=[]
    word = []
    prev_word = 0
    for i in synced_text:
        if i.gt_char_of_word != prev_word:
            if i.gt_char_of_word is not None:
                prev_word = i.gt_char_of_word
                word_list.append(word)
                word = []
        word.append(i)
    word_list.append(word)
    lines = []
    c_line =[]
    c_line_index = 0
    for w in word_list:
        w = [i for i in w if i.char_gt is not None]
        word =  "".join([i.char_gt for i in w if i.char_gt is not None])
        hyphen_word = hyphen.apply_to_word(word)
        syl_gap_char_indexes = [ind for ind, i in enumerate(hyphen_word) if i == "-"]
        syl_gap_char_indexes = [i - ind for ind, i in enumerate(syl_gap_char_indexes)]
        syl_gap_char_indexes.append(len(word))
        prev = 0

        for i in syl_gap_char_indexes:
            syl = w[prev:i]
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

    return doc


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        from omr.steps.text.correction_tools.document_matching_corrector.meta import Meta

        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        self.database_hyphen_dictionary = lyrics_store.syllable_dictionary

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None, gt_book: DatabaseBook = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        single_document_result = []
        all_docs: List[Document] = []
        hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                            right=1)
        MIN_SEARCH_LENGTH = 15
        MAX_ALLOWED_CER = 0.35
        if self.document_id is not None:
            document: Document = documents.database_documents.get_document_by_id(self.document_id)
            all_docs.append(document)
        else:
            for i in pages:
                all_docs += documents.database_documents.get_documents_of_page(i.pcgts().page)
        master_doc_list = documents.database_documents.documents

        for ind_doc, doc in enumerate(all_docs):
            document = doc
            text = document.get_text_of_document(book)
            text_list = document.get_text_list_of_line_document(book)

            original_normalized_text = self.text_normalizer.apply(text)
            search_text = original_normalized_text

            try:
                master_idx = master_doc_list.index(document)
            except ValueError:
                master_idx = -1

            lookahead_idx = master_idx + 1

            while (master_idx != -1 and
                   len(search_text.replace(" ", "")) < MIN_SEARCH_LENGTH and
                   lookahead_idx < len(master_doc_list)):
                next_doc = master_doc_list[lookahead_idx]
                next_doc_text = next_doc.get_text_of_document(book)
                next_doc_text = self.text_normalizer.apply(next_doc_text)

                search_text += " " + next_doc_text
                lookahead_idx += 1

            search_stripped = search_text.replace(" ", "").lower()
            search_len = max(len(search_stripped), 1)

            max_allowed_ed = int(search_len * MAX_ALLOWED_CER)
            lowest_ed = max_allowed_ed + 1
            lyric_info: Lyric_info = None

            if gt_book:
                documents2 = DatabaseBookDocuments().load(gt_book)
                document2 = documents2.database_documents.get_document_by_b_uid(document.get_book_u_id())

                text2 = document2.get_text_of_document(gt_book)
                text2 = self.text_normalizer.apply(text2)
                lyric_info = Lyric_info(latine=text2, id="1", index="1", meta_info="1", meta_infos_extended=None,
                                        variants=None)
            else:
                for b in lyrics_store.lyrics.lyrics:
                    target_stripped = self.text_normalizer.apply(b.latine).replace(" ", "").lower()

                    if len(target_stripped) < (search_len - lowest_ed):
                        continue

                    ed = edlib.align(search_stripped, target_stripped, mode="HW", k=lowest_ed)
                    ed_score = ed["editDistance"]

                    if ed_score != -1 and ed_score < lowest_ed:
                        lowest_ed = ed_score
                        lyric_info = b
                        if lowest_ed == 0:
                            break

            if lyric_info and lowest_ed <= max_allowed_ed:
                try:
                    aligned_text = align_documents2(lyric_info, document, original_normalized_text, text_list,
                                                    self.text_normalizer, hyphen)
                except Exception as e:
                    aligned_text = "\n".join(text_list) + "\n"
            else:
                aligned_text = "\n".join(text_list) + "\n"

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

            percentage = ind_doc / len(all_docs)
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


