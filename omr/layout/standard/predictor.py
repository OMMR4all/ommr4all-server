from omr.layout.predictor import LayoutAnalysisPredictor, PredictorParameters, PredictionType, PredictionResult
from typing import List
from database.file_formats.pcgts import PcGts, TextRegionType, Coords
import numpy as np


class StandardLayoutAnalysisPredictor(LayoutAnalysisPredictor):
    def __init__(self, params: PredictorParameters):
        super().__init__(params)

        from layoutanalysis.segmentation.segmentation import Segmentator, SegmentationSettings
        settings = SegmentationSettings()
        self.segmentator = Segmentator(settings)

    def _predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        def extract_staffs(pcgts: PcGts):
            staffs = []
            for mr in pcgts.page.music_regions:
                for s in mr.staffs:
                    staffs.append([list(sl.coords.points[:, ::-1].astype(int)) for sl in s.staff_lines])
            return staffs

        def p_to_np(polys):
            return [Coords(np.array(p.exterior.coords)) for p in polys]

        for p in self.segmentator.segmentate(
                map(extract_staffs, pcgts_files),
                [p.page.location.file('gray_deskewed').local_path() for p in pcgts_files]):

            yield PredictionResult(
                text_regions={
                    TextRegionType.LYRICS: p_to_np(p.get('lyrics')),
                    TextRegionType.DROP_CAPITAL: p_to_np(p.get('initials')),
                    TextRegionType.PARAGRAPH: p_to_np(p.get('text')),
                },
                music_regions=p_to_np(p.get('system')),
            )


if __name__ == "__main__":
    from database import DatabaseBook
    from PIL import Image
    import matplotlib.pyplot as plt

    b = DatabaseBook('Graduel')
    p = b.page('Graduel_de_leglise_de_Nevers_022')
    img = np.array(Image.open(p.file('color_deskewed').local_path()))
    mask = np.zeros(img.shape, np.float) + 255
    val_pcgts = [PcGts.from_file(p.file('pcgts'))]

    params = PredictorParameters(
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


