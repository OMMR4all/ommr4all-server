import os

from shared.pcgtscanvas import PcGtsCanvas

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from omr.steps.stafflines.detection.predictor import AlgorithmPredictionResultGenerator, PredictionResult, RegionLineMaskData, LineDetectionPredictorCallback, StaffLinePredictor
from database import DatabasePage, DatabaseBook
from database.file_formats.pcgts import *
import numpy as np
import os
import logging
from typing import List, Optional, NamedTuple
from omr.steps.stafflines.detection.dataset import PCDataset
from omr.steps.stafflines.detection.pixelclassifier.meta import Meta, AlgorithmMeta
from linesegmentation.detection.callback import LineDetectionCallback
from linesegmentation.detection.settings import PostProcess
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, PredictionProgress


logger = logging.getLogger(__name__)


# LineDetectionCallback's own default, which is calibrated for detect_fcn() --
# the code path this (legacy, non-torch) predictor actually takes via detect().
STEPS_PER_PAGE = 8


class StaffLinePredictorParameters(NamedTuple):
    post_processing: PostProcess = PostProcess.BESTFIT
    best_fit_scale = 4.0


class PCPredictionCallback(LineDetectionCallback):
    """Bridges the line-detection library's counters onto PredictionProgress.

    detect() -> detect_fcn() drives both update_total_state() and
    update_page_counter(), so the library's page counter is authoritative here
    and we only translate its step counter into a within-page fraction.
    """

    def __init__(self, progress: PredictionProgress):
        super().__init__(steps_per_page=STEPS_PER_PAGE)
        self.progress = progress
        self._pages_done = 0
        self._steps_in_page = 0

    def changed(self):
        processed = self.get_processed_pages()
        if processed > self._pages_done:
            self._pages_done = processed
            self._steps_in_page = 0
            self.progress.page_finished()
            return
        self._steps_in_page = min(self._steps_in_page + 1, STEPS_PER_PAGE)
        self.progress.sub_progress(self._steps_in_page / STEPS_PER_PAGE)


class BasicStaffLinePredictor(StaffLinePredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        params = StaffLinePredictorParameters()

        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            min_lines_per_system=self.params.minNumberOfStaffLines,
            line_number=self.params.maxNumberOfStaffLines,
            horizontal_min_length=6,
            line_interpolation=True,
            line_space_height=self.dataset_params.origin_staff_line_distance,
            target_line_space_height=self.dataset_params.target_staff_line_distance,
            model=os.path.join(settings.model.local_file('model.h5')),
            post_process=params.post_processing,
            best_fit_scale=params.best_fit_scale,
            debug=False,
            debug_model=False,
        )
        self.line_detection = LineDetection(self.settings)

    def predict(self, pages: List[DatabasePage], callback: Optional[LineDetectionPredictorCallback] = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        pc_dataset = PCDataset(pcgts_files, self.dataset_params)
        dataset = pc_dataset.to_line_detection_dataset()
        gray_images = [(255 - data.line_image).astype(np.uint8) for data in dataset]
        progress = PredictionProgress(callback, len(pages))
        progress.start()
        # Always overwrite: the predictor is cached across tasks and LineDetection
        # keeps the callback as a member, so a stale one would keep reporting into
        # a finished task. With callback=None the whole chain is a silent no-op.
        # TODO: Line detection callback of line-detection not as class member variable
        self.line_detection.callback = PCPredictionCallback(progress)
        predictions = self.line_detection.detect(gray_images)
        for i, (data, r) in enumerate(zip(dataset, predictions)):
            rlmd: RegionLineMaskData = data
            page: Page = rlmd.operation.page
            logger.debug("Predicted {}/{}. File {}".format(i + 1, len(dataset), rlmd.operation.page.location.local_path()))
            if len(r) == 0:
                logger.warning('No staff lines detected.')
                yield PredictionResult([], [], rlmd)
            else:
                def transform_points(yx_points):
                    return Coords(np.array([pc_dataset.local_to_global_pos(Point(p[1], p[0]), rlmd.operation.params).p for p in yx_points]))

                ml_global = [Line(staff_lines=StaffLines([StaffLine(page.image_to_page_scale(transform_points(list(pl)), rlmd.operation.scale_reference)) for pl in l])) for l in r]
                ml_local = [Line(staff_lines=StaffLines([StaffLine(page.image_to_page_scale(Coords(np.array(pl)[:, ::-1]), rlmd.operation.scale_reference)) for pl in l])) for l in r]
                yield PredictionResult(ml_global, ml_local, rlmd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    # page = Book('demo').page('page00000001')
    book = DatabaseBook('test')
    page = book.pages()[0]
    # page = book.page('Graduel_de_leglise_de_Nevers_032_rot')  # zacken in linie
    # page = book.page('Graduel_de_leglise_de_Nevers_531')
    # page = book.page('Graduel_de_leglise_de_Nevers_030')
    # page = book.page('Graduel_de_leglise_de_Nevers_520')
    # page = book.page('Graduel_de_leglise_de_Nevers_513')
    pages = [page]

    settings = AlgorithmPredictorSettings(
        Meta.best_model_for_book(book),
        book.get_meta().algorithm_predictor_params(BasicStaffLinePredictor.meta().type()),
        #["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/internal_storage/default_models/french14/pc_staff_lines/model"],
        #["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/models_out/all/line_detection_4/best"],
        # ["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/storage/Graduel/pc_staff_lines/model"],
        # ["/home/wick/Downloads/line_detection_0/best"],
    )
    detector = BasicStaffLinePredictor(settings)
    for prediction in detector.predict(pages):
        canvas = PcGtsCanvas(prediction.line.operation.page, PageScaleReference.NORMALIZED)
        def scale(p):
            return prediction.line.operation.page.page_to_image_scale(p, ref=PageScaleReference.NORMALIZED)

        f, ax = plt.subplots(1, 3)
        staffs = prediction.music_lines_local
        data = prediction.line
        img = np.array(data.line_image, dtype=np.uint8)
        ax[0].imshow(255 - img, cmap='gray')
        [s.draw(img, color=255, thickness=1, scale=scale) for s in staffs]
        b = np.zeros(img.shape)
        [s.draw(b, color=255, thickness=1, scale=scale) for s in staffs]
        ax[1].imshow(img)
        ax[2].imshow(b, cmap='gray')
        plt.show()

