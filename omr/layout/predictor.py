from abc import ABC, abstractmethod
from typing import List, Generator, NamedTuple, Dict, Optional
from database.file_formats.pcgts import *
from omr.symboldetection.dataset import SymbolDetectionDataset
from enum import IntEnum


class LayoutPredictorParameters(NamedTuple):
    checkpoints: List[str]
    page_scale_reference: PageScaleReference = PageScaleReference.NORMALIZED


class PredictionResult(NamedTuple):
    text_regions: Dict[TextRegionType, List[Coords]]
    music_regions: List[Coords]


PredictionType = Generator[PredictionResult, None, None]


class IdCoordsPair(NamedTuple):
    coords: Coords
    id: str = None

    def to_dict(self):
        return {
            'coords': self.coords.to_json(),
            'id': self.id,
        }


class FinalPredictionResult(NamedTuple):
    text_regions: Dict[TextRegionType, List[IdCoordsPair]]
    music_regions: List[IdCoordsPair]

    def to_dict(self):
        return {
            'textRegions': {
                key.value: [v.to_dict() for v in val] for key, val in self.text_regions.items()
            },
            'musicRegions': [v.to_dict() for v in self.music_regions]
        }


FinalPrediction = Generator[FinalPredictionResult, None, None]


class LayoutAnalysisPredictorCallback(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def progress_updated(self, percentage: float):
        pass


class LayoutAnalysisPredictor(ABC):
    def __init__(self, params: LayoutPredictorParameters):
        self.params = params
        self.dataset: SymbolDetectionDataset = None

    def predict(self, pcgts_files: List[PcGts], callback: Optional[LayoutAnalysisPredictorCallback] = None) -> FinalPrediction:
        for r, pcgts in zip(self._predict(pcgts_files, callback=callback), pcgts_files):
            music_lines = []
            for mr in pcgts.page.music_regions:
                music_lines += mr.staffs

            # music lines must be sorted
            music_lines.sort(key=lambda ml: ml.center_y())

            for ml, coords in zip(music_lines, r.music_regions):
                ml.coords = coords

            yield FinalPredictionResult(
                {k: [IdCoordsPair(c) for c in coords] for k, coords in r.text_regions.items()},
                [IdCoordsPair(coords, str(ml.id)) for ml, coords in zip(music_lines, r.music_regions)]
            )

    @abstractmethod
    def _predict(self, pcgts_files: List[PcGts], callback: Optional[LayoutAnalysisPredictorCallback] = None)\
            -> PredictionType:
        pass


class PredictorTypes(IntEnum):
    STANDARD = 0
    LYRICS_BBS = 1


def create_predictor(t: PredictorTypes, params: LayoutPredictorParameters) -> LayoutAnalysisPredictor:
    if t == PredictorTypes.STANDARD:
        from omr.layout.standard.predictor import StandardLayoutAnalysisPredictor
        return StandardLayoutAnalysisPredictor(params)
    elif t == PredictorTypes.LYRICS_BBS:
        from omr.layout.lyricsbbs.predictor import LyricsBBSLayoutAnalysisPredictor
        return LyricsBBSLayoutAnalysisPredictor(params)

    raise Exception('Invalid type {}'.format(type))


if __name__ == "__main__":
    from database import DatabaseBook
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    b = DatabaseBook('demo')
    p = b.page('page00000001')
    img = np.array(Image.open(p.file('color_norm').local_path()))
    mask = np.zeros(img.shape, np.float) + 255
    val_pcgts = [PcGts.from_file(p.file('pcgts'))]

    params = LayoutPredictorParameters(
        checkpoints=[],
    )
    pred = create_predictor(PredictorTypes.LYRICS_BBS, params)

    def s(c):
        return val_pcgts[0].page.page_to_image_scale(c, params.page_scale_reference)

    for p in pred.predict(val_pcgts):
        for i, mr_c in enumerate(p.music_regions):
            s(mr_c.coords).draw(mask, (255, 0, 0), fill=True, thickness=0)

        for i, mr_c in enumerate(p.text_regions.get(TextRegionType.LYRICS, [])):
            s(mr_c.coords).draw(mask, (0, 255, 0), fill=True, thickness=0)

        for i, mr_c in enumerate(p.text_regions.get(TextRegionType.DROP_CAPITAL, [])):
            s(mr_c.coords).draw(mask, (0, 0, 255), fill=True, thickness=0)

    import json
    print(p.to_dict())
    print(json.dumps(p.to_dict()))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(img.mean(axis=-1, keepdims=True).astype(float) * mask.astype(float) / 255 / 255)
    plt.show()


