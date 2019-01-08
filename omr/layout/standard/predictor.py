from omr.layout.predictor import LayoutAnalysisPredictor, PredictorParameters, PredictionType
from typing import List
from omr.datatypes import PcGts


class StandardLayoutAnalysisPredictor(LayoutAnalysisPredictor):
    def __init__(self, params: PredictorParameters):
        super().__init__(params)

        from layoutanalysis.segmentation.segmentation import Segmentator, SegmentationSettings
        settings = SegmentationSettings()
        self.segmentator = Segmentator(settings)

    def predict(self, pcgts_files: List[PcGts]) -> PredictionType:
        def extract_staffs(pcgts: PcGts):
            staffs = []
            for mr in pcgts.page.music_regions:
                for s in mr.staffs:
                    staffs.append([list(sl.coords.points[:, ::-1].astype(int)) for sl in s.staff_lines])
            return staffs

        for p in self.segmentator.segmentate(
                map(extract_staffs, pcgts_files),
                [p.page.location.file('gray_deskewed').local_path() for p in pcgts_files]):

            yield p


if __name__ == "__main__":
    import numpy as np
    import main.book as book
    from PIL import Image
    import matplotlib.pyplot as plt
    from layoutanalysis.segmentation.segmentation import draw_polygons

    b = book.Book('Graduel')
    p = b.page('Graduel_de_leglise_de_Nevers_022')
    img = np.array(Image.open(p.file('gray_deskewed').local_path()))
    val_pcgts = [PcGts.from_file(p.file('pcgts'))]

    params = PredictorParameters(
        checkpoints=[],
    )
    pred = StandardLayoutAnalysisPredictor(params)
    for p in pred.predict(val_pcgts):
        p: dict = p
        for i, k in enumerate(p.keys()):
            draw_polygons(p[k], img)

    plt.imshow(img)
    plt.show()


