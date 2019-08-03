from abc import abstractmethod
from typing import List, Generator, NamedTuple, Dict, Optional
from database.file_formats.pcgts import *
from database import DatabasePage
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.steps.algorithm import AlgorithmPredictor, PredictionCallback, AlgorithmPredictorParams, AlgorithmPredictorSettings, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator
from database.file_formats.pcgts import PageScaleReference
import logging

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    blocks: Dict[BlockType, List[Coords]]


PredictionType = Generator[PredictionResult, None, None]


class IdCoordsPair(NamedTuple):
    coords: Coords
    id: str = None

    def to_dict(self):
        return {
            'coords': self.coords.to_json(),
            'id': self.id,
        }


class FinalPredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class FinalPredictionResult(NamedTuple, AlgorithmPredictionResult, metaclass=FinalPredictionResultMeta):
    blocks: Dict[BlockType, List[IdCoordsPair]]
    pcgts: PcGts

    def to_dict(self):
        return {
            'blocks': {
                key.value: [v.to_dict() for v in val] for key, val in self.blocks.items()
            }
        }

    def store_to_page(self):
        pcgts = self.pcgts
        page = self.pcgts.dataset_page()
        pcgts.page.clear_text_blocks()
        for type, id_coords in self.blocks.items():
            for ic in id_coords:
                if type == BlockType.MUSIC:
                    ml = pcgts.page.music_line_by_id(ic.id)
                    if not ml:
                        logger.warning('Music line with id "{}" not found'.format(ic.id))
                        continue

                    ml.coords = ic.coords
                else:
                    pcgts.page.blocks.append(
                        Block(type, ic.id, lines=[Line(coords=ic.coords)])
                    )

        pcgts.to_file(page.file('pcgts').local_path())


class LayoutAnalysisPredictor(AlgorithmPredictor):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return len(page.pcgts().page.text_blocks()) == 0

    def predict(self, pages: List[DatabasePage], callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        for r, pcgts in zip(self._predict(pcgts_files, callback=callback), pcgts_files):
            music_lines = pcgts.page.all_music_lines()

            # music lines must be sorted
            music_lines.sort(key=lambda ml: ml.center_y())

            for ml, coords in zip(music_lines, r.blocks.get(BlockType.MUSIC, [])):
                ml.coords = coords

            yield FinalPredictionResult(
                blocks={
                    **{
                        k: [IdCoordsPair(c) for c in coords] for k, coords in r.blocks.items() if k != BlockType.MUSIC
                    },
                    **{
                        BlockType.MUSIC: [IdCoordsPair(coords, str(ml.id)) for ml, coords in zip(music_lines, r.blocks[BlockType.MUSIC])],
                    }
                },
                pcgts=pcgts
            )

    @abstractmethod
    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None)\
            -> PredictionType:
        pass


if __name__ == "__main__":
    from database import DatabaseBook
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    from omr.steps.step import Step, AlgorithmTypes

    b = DatabaseBook('demo')
    p = b.page('page00000001')
    img = np.array(Image.open(p.file('color_norm').local_path()))
    mask = np.zeros(img.shape, np.float) + 255
    val_pcgts = [PcGts.from_file(p.file('pcgts'))]

    settings = PredictorSettings()
    pred = Step.create_predictor(AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES, settings)

    def s(c):
        return val_pcgts[0].page.page_to_image_scale(c, settings.page_scale_reference)

    for p in pred.predict(val_pcgts):
        for i, mr_c in enumerate(p.blocks.get(BlockType.MUSIC, [])):
            s(mr_c.coords).draw(mask, (255, 0, 0), fill=True, thickness=0)

        for i, mr_c in enumerate(p.blocks.get(BlockType.LYRICS, [])):
            s(mr_c.coords).draw(mask, (0, 255, 0), fill=True, thickness=0)

        for i, mr_c in enumerate(p.blocks.get(BlockType.DROP_CAPITAL, [])):
            s(mr_c.coords).draw(mask, (0, 0, 255), fill=True, thickness=0)

    import json
    print(p.to_dict())
    print(json.dumps(p.to_dict()))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(img.mean(axis=-1, keepdims=True).astype(float) * mask.astype(float) / 255 / 255)
    plt.show()


