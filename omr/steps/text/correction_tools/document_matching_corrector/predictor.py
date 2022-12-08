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

from ...hyphenation.hyphenator import Pyphenator, CombinedHyphenator


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
                #self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


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
        if self.database_hyphen_dictionary is None:
            db = DatabaseDictionary.load(book=book)
            self.database_hyphen_dictionary = db.to_hyphen_dict()
        hyphen = CombinedHyphenator(lang="la_classic", left=2, right=2, dictionary=self.database_hyphen_dictionary)

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
            for b in lyrics.lyrics:
                b: Lyric_info = b
                text2 = b.latine.lower().replace("[", "").replace("]", "")
                ed = edlib.align(text.replace(" ", ""), text2.replace(" ", ""), mode="SHW", k=lowest_ed)
                if 0 < ed["editDistance"] < lowest_ed:
                    lowest_ed = ed["editDistance"]
                    lowest_text = text2

            text_list = document.get_text_list_of_line_document(book)
            text = text.replace(" ", "")
            for i in [lowest_text]:
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
                    locations = ed["locations"][0]
                    i = i[locations[0]:locations[1]]
                    ed = edlib.align(text, i, mode="NW", task="path")
                cigar = ed["cigar"]
                a = edlib.getNiceAlignment(ed, text, i)
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
                for i in aligned_text.split("\n"):
                    i = hyphen.apply_to_sentence(i)
                    aligned_text_elements.append(i)
                aligned_text = "\n".join(aligned_text_elements)
                single_document_result.append(SingleDocumentResult(matched_document=aligned_text,
                                                                   page_id=document.start.page_id,
                                                                   document_id=document.doc_id,
                                                                   document=document,
                                                                   book=book)) #.append(aligned_text)
            if callback:
                callback.progress_updated(ind_doc / len(all_docs))
            # for key, count in count.most_common(5):
            #    #print(self.document_similar_tester.document_dict[key].sentence)
            #    #print(self.document_similar_tester.document_dict[key].get_word_list())
            #    #print(self.document_similar_tester.document_dict[key].get_text())
            #    texts.append(self.document_similar_tester.document_dict[key].get_text())

        yield Result(single_document_result)
