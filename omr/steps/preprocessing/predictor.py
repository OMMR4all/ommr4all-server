from database import DatabaseBook, DatabasePage
from database.file_formats.pcgts import *
import logging
from typing import List, Optional, Tuple
from omr.steps.preprocessing.meta import Meta
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorSettings, AlgorithmPredictorParams, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator
import multiprocessing


logger = logging.getLogger(__name__)


files = ['color_highres_preproc', 'color_lowres_preproc', 'color_norm', 'color_norm_x2', 'connected_components_norm']# 'gray_norm', 'gray_norm_x2', 'binary_norm_x2', 'binary_norm']


def _process_single(args: Tuple[DatabasePage, AlgorithmPredictorParams]):
    page, settings = args

    # update page meta
    meta = page.meta()
    meta.preprocessing.average_line_distance = settings.avgLd
    meta.preprocessing.auto_line_distance = settings.automaticLd
    meta.preprocessing.deskew = settings.deskew

    meta.save(page)

    # process all files
    for file in files:
        # create or recreate files
        file = page.file(file)
        file.delete()
        file.create()


class PreprocessingResult(AlgorithmPredictionResult):
    def to_dict(self):
        return {}

    def store_to_page(self):
        pass


class PreprocessingPredictor(AlgorithmPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        if callback:
            callback.progress_updated(0, len(pages), 0)

        with multiprocessing.Pool(processes=4) as pool:
            for i, _ in enumerate(pool.imap_unordered(_process_single, [(p, self.params) for p in pages])):
                percentage = (i + 1) / len(pages)
                if callback:
                    callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(pages))
                yield PreprocessingResult()

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return any([not page.file(f).exists() for f in files])
