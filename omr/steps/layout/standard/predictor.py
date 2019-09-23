from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings
from typing import List, Optional
from database.file_formats.pcgts import PcGts, Coords, BlockType
import numpy as np
from layoutanalysis.segmentation.callback import SegmentationCallback
from omr.steps.layout.standard.meta import Meta


class SPredictionCallback(SegmentationCallback):
    def __init__(self, callback: PredictionCallback):
        self.callback = callback
        super().__init__()

    def changed(self):
        self.callback.progress_updated(self.get_current_page_progress())


class Predictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        from layoutanalysis.segmentation.segmentation import Segmentator, SegmentationSettings
        self.segmentator = Segmentator(SegmentationSettings())

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        def extract_staffs(pcgts: PcGts):
            page = pcgts.page
            staffs = []
            for mr in pcgts.page.music_blocks():
                for s in mr.lines:
                    staffs.append([list(page.page_to_image_scale(sl.coords, self.dataset_params.page_scale_reference).points[:, ::-1].astype(int)) for sl in s.staff_lines])
            return staffs

        def p_to_np(polys, page):
            return [page.image_to_page_scale(Coords(np.array(p.exterior.coords)), self.dataset_params.page_scale_reference) for p in polys]
        if callback:
            # TODO: Layout analyse callback of layout-analyse not as class member variable
            self.segmentator.callback = SPredictionCallback(callback)

        for p, pcgts in zip(self.segmentator.segment(
                map(extract_staffs, pcgts_files),
                [p.page.location.file(self.dataset_params.page_scale_reference.file('gray'), True).local_path() for p in pcgts_files], ), pcgts_files):
            page = pcgts.page

            yield PredictionResult(
                blocks={
                    BlockType.LYRICS: p_to_np(p.get('lyrics', []), page),
                    BlockType.DROP_CAPITAL: p_to_np(p.get('initials', []), page),
                    BlockType.PARAGRAPH: p_to_np(p.get('text', []), page),
                    BlockType.MUSIC: p_to_np(p.get('system', []), page),
                },
            )


if __name__ == "__main__":
    from database import DatabaseBook
    from PIL import Image
    import matplotlib.pyplot as plt

    b = DatabaseBook('demo')
    p = b.page('page00000001')
    img = np.array(Image.open(p.file('color_norm').local_path()))
    mask = np.zeros(img.shape, np.float) + 255
    val_pcgts = [PcGts.from_file(p.file('pcgts'))]

    params = AlgorithmPredictorSettings(
        model=Meta.best_model_for_book(b),
    )
    pred = Predictor(params)
    def s(c):
        return val_pcgts[0].page.page_to_image_scale(c, pred.dataset_params.page_scale_reference)

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


