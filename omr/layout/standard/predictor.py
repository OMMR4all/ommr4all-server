from omr.layout.predictor import LayoutAnalysisPredictor, LayoutPredictorParameters, PredictionType, PredictionResult\
    , LayoutAnalysisPredictorCallback
from typing import List, Optional
from database.file_formats.pcgts import PcGts, TextRegionType, Coords
import numpy as np
from layoutanalysis.segmentation.callback import SegmentationCallback


class SPredictionCallback(SegmentationCallback):
    def __init__(self, callback: LayoutAnalysisPredictorCallback):
        self.callback = callback
        super().__init__()

    def changed(self):
        self.callback.progress_updated(self.get_current_page_progress())


class StandardLayoutAnalysisPredictor(LayoutAnalysisPredictor):
    def __init__(self, params: LayoutPredictorParameters):
        super().__init__(params)

        from layoutanalysis.segmentation.segmentation import Segmentator, SegmentationSettings
        settings = SegmentationSettings()
        self.segmentator = Segmentator(settings)

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[LayoutAnalysisPredictorCallback] = None) -> PredictionType:
        def extract_staffs(pcgts: PcGts):
            page = pcgts.page
            staffs = []
            for mr in pcgts.page.music_regions:
                for s in mr.staffs:
                    staffs.append([list(page.page_to_image_scale(sl.coords, self.params.page_scale_reference).points[:, ::-1].astype(int)) for sl in s.staff_lines])
            return staffs

        def p_to_np(polys, page):
            return [page.image_to_page_scale(Coords(np.array(p.exterior.coords)), self.params.page_scale_reference) for p in polys]
        if callback:
            # TODO: Layout analyse callback of layout-analyse not as class member variable
            self.segmentator.callback = SPredictionCallback(callback)

        for p, pcgts in zip(self.segmentator.segment(
                map(extract_staffs, pcgts_files),
                [p.page.location.file(self.params.page_scale_reference.file('gray'), True).local_path() for p in pcgts_files], ), pcgts_files):
            page = pcgts.page

            yield PredictionResult(
                text_regions={
                    TextRegionType.LYRICS: p_to_np(p.get('lyrics', []), page),
                    TextRegionType.DROP_CAPITAL: p_to_np(p.get('initials', []), page),
                    TextRegionType.PARAGRAPH: p_to_np(p.get('text', []), page),
                },
                music_regions=p_to_np(p.get('system', []), page),
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

    params = LayoutPredictorParameters(
        checkpoints=[],
    )
    pred = StandardLayoutAnalysisPredictor(params)
    for p in pred.predict(val_pcgts):
        for i, mr_c in enumerate(p.music_regions):
            mr_c.coords.draw(mask, (255, 0, 0), fill=True)

        for i, mr_c in enumerate(p.text_regions.get(TextRegionType.LYRICS, [])):
            mr_c.coords.draw(mask, (0, 255, 0), fill=True)

        for i, mr_c in enumerate(p.text_regions.get(TextRegionType.DROP_CAPITAL, [])):
            mr_c.coords.draw(mask, (0, 0, 255), fill=True)

    import json
    print(p.to_dict())
    print(json.dumps(p.to_dict()))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(img.mean(axis=-1, keepdims=True).astype(float) * mask.astype(float) / 255 / 255)
    plt.show()


