import difflib
from difflib import SequenceMatcher

import numpy as np
from calamari_ocr.proto import CTCDecoderParams
from prettytable import PrettyTable
from unidecode import unidecode

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
from omr.steps.text.predictor import PredictionResult as TextPredictionResult


class ResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):

    def to_dict(self):
        return {}


    def store_to_page(self):
        pass


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.document_id = settings.params.documentId
        self.document_text = settings.params.documentText

        self.document_similar_tester = SimilarDocumentChecker()
        self.text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
        meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
        from ommr4all.settings import BASE_DIR
        model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/fraktur/text_calamari/', meta.type()))
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        settings.params.ctcDecoder.params.type = CTCDecoderParams.CTC_DEFAULT
        self.ocr_predictor = meta.create_predictor(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(self.document_id)
        pages = [DatabasePage(book, x) for x in document.pages_names]
        # Todo Not Implemented yet
        yield Result()


        # for now just assign text based on syllables

        #text = document.get_text_of_document(book)
        #for page in document.pages_names:
        #for i, r in enumerate(self.ocr_predictor.predict(pages)):
        #    ocr_r: TextPredictionResult = r
        #    #
        #    #match_r = [self.match_text(text_line_r) for text_line_r in ocr_r.text_lines if len(text_line_r.line.operation.text_line.sentence.syllables) > 0]
        #corrected_text = self.document_text
        #sm = SequenceMatcher(a=text, b=corrected_text, autojunk=False, isjunk=False)
        #codes = sm.get_opcodes()
        #for opt_code in codes:
        #    if opt_code[0] == 'equal':
        #        print(text[opt_code[1]: opt_code[2]])
        #fromlines = open(text, 'U').readlines()
        #tolines = open(corrected_text, 'U').readlines()
        #diff = difflib.HtmlDiff().make_file(text, corrected_text, "from", "to")
        #print(diff)
        #print(text)
        #print(corrected_text)
        #print(sm.get_opcodes())
        #print(document.doc_id)
        #print(self.document_text)
        #if self.document_id:
        #yield Result(texts)

