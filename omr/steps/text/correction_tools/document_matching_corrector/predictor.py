from calamari_ocr.proto import CTCDecoderParams

from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.book.document import Document
from database.model import Model, MetaId
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage
from typing import List, Optional, NamedTuple

from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step
from omr.text_matching.populate_db import SimilarDocumentChecker
from .meta import Meta
from database.file_formats.pcgts import Coords, PageScaleReference


class ResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    documents: List[str]

    def to_dict(self):
        return {'similarText': [s for s in self.documents]}


    def store_to_page(self):
        pass


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        self.document_similar_tester = SimilarDocumentChecker()
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(self.document_id)
        text = document.get_text_of_document(book)
        text = self.text_normalizer.apply(text)
        text = text.split(' ')
        count = self.document_similar_tester.check_word_based_similarity(text)
        texts = []
        for key, count in count.most_common(5):
            #print(self.document_similar_tester.document_dict[key].sentence)
            #print(self.document_similar_tester.document_dict[key].get_word_list())
            #print(self.document_similar_tester.document_dict[key].get_text())
            texts.append(self.document_similar_tester.document_dict[key].get_text())

        yield Result(texts)

