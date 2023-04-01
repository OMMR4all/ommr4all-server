import math
import os
import re

import edlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from nautilus_ocr.decoder import DecoderOutput, DecoderType
from nautilus_ocr.predict import Network

from database.database_book_documents import DatabaseBookDocuments
from database.database_dictionary import DatabaseDictionary
from database.file_formats.book.document import Document
from database.file_formats.pcgts.page import Sentence
from database.model import Model, MetaId
from ommr4all.settings import BASE_DIR
from omr.dataset import RegionLineMaskData
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage, DatabaseBook
from typing import List, Optional, NamedTuple, Tuple

from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.step import Step
from omr.text_matching.populate_db import SimilarDocumentChecker
from tools.simple_gregorianik_text_export import Lyric_info
from omr.steps.text.text_localisation.meta import Meta
from database.file_formats.pcgts import Coords, PageScaleReference, Point
from database.start_up.load_text_variants_in_memory import lyrics

from itertools import zip_longest
from omr.steps.text.predictor import PredictionResult as TextPredictionResult

from omr.steps.text.dataset import TextDataset
from nautilus_ocr.predict import Network, get_config


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

        meta = Step.meta(AlgorithmTypes.OCR_NAUTILUS)
        from ommr4all.settings import BASE_DIR
        model = Model(
            MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/french14/text_nautilus/', meta.type()))
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        self.ocr_predictor = meta.create_predictor(settings)

    def _predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) :
        if callback:
            callback.progress_updated(0, len(pages), 0)
        unicode_font = ImageFont.truetype("/home/alexanderh/PycharmProjects/pythonProject4/font/GregorianFLF.ttf", 35)

        for i, r in enumerate(self.ocr_predictor.predict(pages, callback=callback)):
            ocr_r: TextPredictionResult = r
            page = ocr_r.dataset_page.pcgts().page
            scale_reference = PageScaleReference.HIGHRES

            img = Image.open(page.location.file(scale_reference.file('color')).local_path())
            draw = ImageDraw.Draw(img)

            for a in ocr_r.text_lines:
                for z in a.chars:
                    print(z[0])
                    for t in z[1]:
                        print(t.x)
                        print(ocr_r.dataset_page.pcgts().page.page_to_image_scale(t.x, scale_reference))
                    line = [ (ocr_r.dataset_page.pcgts().page.page_to_image_scale(x.x, scale_reference),
                              ocr_r.dataset_page.pcgts().page.page_to_image_scale(x.y, scale_reference)) for x in z[1]]
                    draw.line(line)
                    draw.text(line[0], z[0], font=unicode_font, fill='white')
            plt.imshow(np.array(img))
            plt.show()

            percentage = (i + 1) / len(pages)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        if not page.pcgts():
            return True
        return len(page.pcgts().page.annotations.connections) == 0

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        self._predict(pages, callback)



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
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True),
                                                               LockState(Locks.LAYOUT, True)], True, [
                                                             # DatabaseBook('Graduel_Part_1'),
                                                             # DatabaseBook('Graduel_Part_2'),
                                                             # DatabaseBook('Graduel_Part_3'),
                                                         ])
    book = DatabaseBook('Graduel_Part_1')
    meta = Step.meta(AlgorithmTypes.TEXT_LOCALISATION)
    # model = meta.newest_model_for_book(book)
    # model = Model(
    #    MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical',
    ##                            meta.type()))
    #meta2 = Step.meta(AlgorithmTypes.OCR_NAUTILUS)

    pred = Predictor(AlgorithmPredictorSettings(Meta.best_model_for_book(book)))
    ps: List[PredictionResult] = list(pred.predict(book.pages()[0:1]))
    for i, p in enumerate(ps):
        canvas = PcGtsCanvas(p.pcgts.page, p.text_lines[0].line.operation.scale_reference)
        for j, s in enumerate(p.text_lines):
            canvas.draw(s)

        canvas.show()
