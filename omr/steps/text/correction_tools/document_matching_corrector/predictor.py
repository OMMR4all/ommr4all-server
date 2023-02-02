import math
import re

import edlib

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
from omr.steps.step import Step
from omr.text_matching.populate_db import SimilarDocumentChecker
from tools.simple_gregorianik_text_export import Lyric_info
from .meta import Meta
from database.file_formats.pcgts import Coords, PageScaleReference
from database.start_up.load_text_variants_in_memory import lyrics

from itertools import zip_longest

from ...hyphenation.hyphenator import Pyphenator, CombinedHyphenator, HyphenDicts, Hyphenator


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

    def to_dict(self):
        return {'similarText': self.matched_document,
                "page_id": self.page_id,
                "document_id": self.document_id}


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    documents: List[SingleDocumentResult]

    def to_dict(self):
        return {'docs': [s.to_dict() for s in self.documents]}

    def store_to_page(self):
        for doc in self.documents:
            lines = doc.document.get_page_line_of_document(book=doc.book)
            matched_lines = doc.matched_document.split("\n")
            for line_ind, line in enumerate(lines):
                line, page = line
                if line_ind < len(matched_lines):
                    line.sentence = Sentence.from_string(matched_lines[line_ind])
                    page.pcgts().page.annotations.connections.clear()
                    page.pcgts().to_file(page.file('pcgts').local_path())
                # self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        # self.document_similar_tester = SimilarDocumentChecker()
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        self.database_hyphen_dictionary = None

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        single_document_result = []
        all_docs: List[Document] = []
        # if self.database_hyphen_dictionary is None:
        #    db = DatabaseDictionary.load(book=book)
        #    self.database_hyphen_dictionary = db.to_hyphen_dict()
        hyphen = Pyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                            right=1)  # dictionary=self.database_hyphen_dictionary)

        if self.document_id is not None:
            document: Document = documents.database_documents.get_document_by_id(self.document_id)
            all_docs.append(document)
        else:
            for i in pages:
                all_docs += documents.database_documents.get_documents_of_page(i.pcgts().page)
        for ind_doc, doc in enumerate(all_docs):
            document = doc
            text = document.get_text_of_document(book)

            text = self.text_normalizer.apply(text)
            lowest_ed = 999999
            lowest_text = ""
            ed = edlib.align("abc", "abc", mode="SHW", k=lowest_ed)

            for b in lyrics.lyrics:
                b: Lyric_info = b
                text2 = b.latine.lower().replace("[", "").replace("]", "")
                ed = edlib.align(text.replace(" ", ""), text2.replace(" ", ""), mode="SHW", k=lowest_ed)
                if 0 < ed["editDistance"] < lowest_ed:
                    lowest_ed = ed["editDistance"]
                    lowest_text = text2
                elif text.replace(" ", "") == text2.replace(" ", ""):
                    lowest_ed = 0
                    lowest_text = text2

            # print((len(lowest_text.replace(" ", "")) - lowest_ed) / len(lowest_text.replace(" ", "")))
            text_list = document.get_text_list_of_line_document(book)
            # text_whitespaces = [ind for ind, i in enumerate(text) if i == " "]
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
                        print(ed)
                        print(text)
                        print(i)
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
                    aligned_text_elements = []

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
                                            lines[line_ind] = lines[line_ind][:-len(last_word)] + last_word + words_l2[0]
                                            lines[line_ind + 1] = lines[line_ind + 1][len(words_l2[0]):] #.replace(words_l2[0], "")
                                            break
                                else:
                                    stop_while = True
                                    break
                        lines = [i.strip() for i in lines if len(i) > 0]
                        a_text = "\n".join(lines)
                        return a_text
                    aligned_text = correct_line_splits(aligned_text, orig_text)
                    for i in aligned_text.split("\n"):
                        i = hyphen.apply_to_sentence(i)
                        aligned_text_elements.append(i)
                    aligned_text = "\n".join(aligned_text_elements)

                    single_document_result.append(SingleDocumentResult(matched_document=aligned_text,
                                                                       page_id=document.start.page_id,
                                                                       document_id=document.doc_id,
                                                                       document=document,
                                                                       book=book))  # .append(aligned_text)

            percentage = (ind_doc) / len(all_docs)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=ind_doc + 1, n_pages=len(all_docs))

        yield Result(single_document_result)
