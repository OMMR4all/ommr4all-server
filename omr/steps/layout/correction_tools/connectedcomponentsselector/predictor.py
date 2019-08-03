from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, PredictionCallback, AlgorithmPredictionResultGenerator
from database import DatabasePage
from typing import List, Optional, NamedTuple
from .connected_component_selector import extract_components
from .meta import Meta
from database.file_formats.pcgts import Coords, PageScaleReference


class ResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class Result(NamedTuple, AlgorithmPredictionResult, metaclass=ResultMeta):
    polys: List[Coords]

    def to_dict(self):
        return {
            'polys': [p.to_json() for p in self.polys]
        }

    def store_to_page(self):
        pass


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.initial_line = settings.params.initialLine

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        for page in pages:
            yield self.predict_single(page)

    def predict_single(self, page: DatabasePage) -> Result:
        pcgts = page.pcgts()
        import pickle
        staff_lines: List[Coords] = []
        pcgts = pcgts
        for mr in pcgts.page.music_blocks():
            for ml in mr.lines:
                staff_lines += [pcgts.page.page_to_image_scale(s.coords, PageScaleReference.NORMALIZED) for s in ml.staff_lines]

        with open(page.file('connected_components_norm', create_if_not_existing=True).local_path(), 'rb') as pkl:
            polys = extract_components(pickle.load(pkl), pcgts.page.page_to_image_scale(self.initial_line, PageScaleReference.NORMALIZED), staff_lines)
            polys = [pcgts.page.image_to_page_scale(c, PageScaleReference.NORMALIZED) for c in polys]

        return Result(polys)
