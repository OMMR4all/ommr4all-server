import os

from segmentation.model_builder import ModelBuilderLoad
from segmentation.network import EnsemblePredictor
from segmentation.network_postprocessor import NetworkMaskPostProcessor, MaskPredictionResult
from segmentation.preprocessing.source_image import SourceImage
from segmentation.scripts.train import get_default_device
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
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings


logger = logging.getLogger(__name__)


class StaffLinePredictorParameters(NamedTuple):
    post_processing: PostProcess = PostProcess.BESTFIT
    best_fit_scale = 4.0


class PCPredictionCallback(LineDetectionCallback):
    def __init__(self, callback: LineDetectionPredictorCallback):
        self.callback = callback
        super().__init__()

    def changed(self):
        self.callback.progress_updated(
            self.get_progress(),
            self.get_total_pages(),
            self.get_processed_pages(),
        )

class PCPredictionCallback(LineDetectionCallback):
    def __init__(self, callback: LineDetectionPredictorCallback):
        self.callback = callback
        super().__init__()

    def changed(self):
        self.callback.progress_updated(
            self.get_progress(),
            self.get_total_pages(),
            self.get_processed_pages(),
        )
class BasicStaffLinePredictorTorch(StaffLinePredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

        params = StaffLinePredictorParameters()
        modelbuilder = ModelBuilderLoad.from_disk(model_weights=os.path.join(settings.model.local_file('best.torch')),
                                                  device=get_default_device())

        base_model = modelbuilder.get_model()
        config = modelbuilder.get_model_configuration()
        preprocessing_settings = modelbuilder.get_model_configuration().preprocessing_settings
        self.predictor = EnsemblePredictor([base_model], [preprocessing_settings])
        self.nmaskpredictor = NetworkMaskPostProcessor(self.predictor, config.color_map)
        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            min_lines_per_system=self.params.minNumberOfStaffLines,
            line_number=self.params.maxNumberOfStaffLines,
            horizontal_min_length=6,
            line_interpolation=True,
            line_space_height=self.dataset_params.origin_staff_line_distance,
            target_line_space_height=self.dataset_params.target_staff_line_distance,
            post_process=params.post_processing,
            best_fit_scale=params.best_fit_scale,
            debug=False,
            debug_model=False,
        )
        self.line_detection = LineDetection(self.settings)

    def predict(self, pages: List[DatabasePage], callback: Optional[LineDetectionPredictorCallback] = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        pc_dataset = PCDatasetTorch(pcgts_files, self.dataset_params)
        dataset = pc_dataset.to_line_detection_dataset()

        for ind, i in enumerate(tqdm(dataset, total=len(pages))):
            output: MaskPredictionResult = self.nmaskpredictor.predict_image(SourceImage.from_numpy(i.line_image))
            from scipy.special import softmax
            prob_map_softmax = softmax(output.prediction_result.probability_map, axis=-1)
            #output.generated_mask.show()
            r = self.line_detection.detect_prob_map(output.prediction_result.source_image.get_grayscale_array(), prob_map_softmax)

            rlmd: RegionLineMaskData = i
            page: Page = rlmd.operation.page
            if callback:
                percentage = (ind + 1) / len(pages)

                callback.progress_updated(percentage, n_processed_pages=ind + 1, n_pages=len(pages))
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

