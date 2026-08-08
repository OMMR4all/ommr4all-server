import os

from tqdm import tqdm
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
from omr.steps.stafflines.detection.dataset import PCDataset, PCDatasetTorch
from omr.steps.stafflines.detection.pixelclassifier_torch.meta import Meta, AlgorithmMeta
from linesegmentation.detection.callback import LineDetectionCallback
from linesegmentation.detection.settings import PostProcess
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, PredictionProgress


logger = logging.getLogger(__name__)


# Number of LineDetectionCallback.update_total_state() calls this predictor's code
# path spends per page. It calls LineDetection.detect_prob_map(), which reaches
# detect_staff_lines() and fires 3 unconditional states plus 2 in the BESTFIT
# post-processing branch (see linesegmentation/detection/detection.py). The
# library's own default of 8 is calibrated for detect_fcn(), which we do not use,
# so relying on it would cap our progress at 5/8.
STEPS_PER_PAGE = 5


class StaffLinePredictorParameters(NamedTuple):
    post_processing: PostProcess = PostProcess.BESTFIT
    best_fit_scale = 4.0


class PCPredictionCallback(LineDetectionCallback):
    """Feeds the line-detection library's step counter into PredictionProgress.

    The library counts steps globally across the whole batch; we only care about
    the fraction *within the current page*, because PredictionProgress owns the
    page-level accounting. Reporting the library's global percentage here as well
    is what made the bar oscillate: two emitters on different scales, last write
    wins.
    """

    def __init__(self, progress: PredictionProgress):
        super().__init__(steps_per_page=STEPS_PER_PAGE)
        self.progress = progress
        self._steps_in_page = 0

    def changed(self):
        self._steps_in_page = min(self._steps_in_page + 1, STEPS_PER_PAGE)
        self.progress.sub_progress(self._steps_in_page / STEPS_PER_PAGE)

    def page_finished(self):
        self._steps_in_page = 0
        self.progress.page_finished()


class BasicStaffLinePredictorTorch(StaffLinePredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        from segmentation.model_builder import ModelBuilderLoad
        from segmentation.network import EnsemblePredictor
        from segmentation.network_postprocessor import NetworkMaskPostProcessor
        from segmentation.preprocessing.source_image import SourceImage
        from segmentation.scripts.train import get_default_device
        super().__init__(settings)

        params = StaffLinePredictorParameters()
        device = get_default_device()
        logger.info(f"Using device: {device}")
        modelbuilder = ModelBuilderLoad.from_disk(model_weights=os.path.join(settings.model.local_file('best.torch')),
                                                  device=device)

        base_model = modelbuilder.get_model()
        config = modelbuilder.get_model_configuration()
        print(f"PREDICTOR INIT: model path={settings.model.path}, classes={config.network_settings.classes if config.network_settings else 'N/A'}, color_map entries={len(config.color_map.class_spec) if config.color_map else 'N/A'}")
        preprocessing_settings = modelbuilder.get_model_configuration().preprocessing_settings
        self.predictor = EnsemblePredictor([base_model], [preprocessing_settings])
        self.nmaskpredictor = NetworkMaskPostProcessor(self.predictor, config.color_map)
        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.line_settings = LineDetectionSettings(
            min_lines_per_system=self.params.minNumberOfStaffLines,
            line_number=self.params.maxNumberOfStaffLines,
            horizontal_min_length=6,
            line_interpolation=True,
            line_space_height=self.dataset_params.origin_staff_line_distance,
            target_line_space_height=self.dataset_params.target_staff_line_distance,
            post_process=params.post_processing,
            best_fit_scale=params.best_fit_scale,
            #debug=True,
            #debug_model=True,
        )
        self.line_detection = LineDetection(self.line_settings)

    def predict(self, pages: List[DatabasePage], callback: Optional[LineDetectionPredictorCallback] = None) -> AlgorithmPredictionResultGenerator:
        from segmentation.preprocessing.source_image import SourceImage
        pcgts_files = [p.pcgts() for p in pages]
        pc_dataset = PCDatasetTorch(pcgts_files, self.dataset_params)
        dataset = pc_dataset.to_line_detection_dataset()

        progress = PredictionProgress(callback, len(pages))
        progress.start()

        # The predictor instance is cached across tasks (see predictorcache), and
        # LineDetection keeps the callback as a member. Always overwrite it, or a
        # finished task's callback keeps firing into its dead communication queue.
        # With callback=None the whole chain is a silent no-op.
        # TODO: Line detection callback of line-detection not as class member variable
        ld_callback = PCPredictionCallback(progress)
        ld_callback.set_total_pages(len(pages))
        self.line_detection.callback = ld_callback

        for ind, i in enumerate(tqdm(dataset, total=len(pages))):
            output = self.nmaskpredictor.predict_image(SourceImage.from_numpy(i.line_image))
            from scipy.special import softmax
            prob_map_softmax = softmax(output.prediction_result.probability_map, axis=-1)
            print(f"PREDICTOR DEBUG: prob_map_softmax shape={prob_map_softmax.shape}, line_class mean={prob_map_softmax[:,:,1].mean():.4f}, line_class max={prob_map_softmax[:,:,1].max():.4f}")
            #from matplotlib import pyplot as plt
            #plt.imshow(output.generated_mask)
            #plt.show()
            #output.generated_mask.show()

            r = self.line_detection.detect_prob_map(output.prediction_result.source_image.get_grayscale_array(), prob_map_softmax)

            rlmd: RegionLineMaskData = i
            page: Page = rlmd.operation.page
            ld_callback.page_finished()
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
    book = DatabaseBook('pa_904')
    page = book.pages()[20]
    # page = book.page('Graduel_de_leglise_de_Nevers_032_rot')  # zacken in linie
    # page = book.page('Graduel_de_leglise_de_Nevers_531')
    # page = book.page('Graduel_de_leglise_de_Nevers_030')
    # page = book.page('Graduel_de_leglise_de_Nevers_520')
    # page = book.page('Graduel_de_leglise_de_Nevers_513')
    pages = [page]

    settings = AlgorithmPredictorSettings(
        Meta.best_model_for_book(book),
        book.get_meta().algorithm_predictor_params(BasicStaffLinePredictorTorch.meta().type()),
        #["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/internal_storage/default_models/french14/pc_staff_lines/model"],
        #["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/models_out/all/line_detection_4/best"],
        # ["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/storage/Graduel/pc_staff_lines/model"],
        # ["/home/wick/Downloads/line_detection_0/best"],
    )
    detector = BasicStaffLinePredictorTorch(settings)
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

