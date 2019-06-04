from omr.stafflines.detection.predictor import StaffLinesPredictor, StaffLinePredictorParameters, PredictionType, PredictionResult, RegionLineMaskData, StaffLineDetectionDatasetParams, LineDetectionPredictorCallback
from database import DatabasePage, DatabaseBook
from database.file_formats.pcgts import *
import numpy as np
import os
import logging
from typing import List, Optional
import omr.stafflines.detection.pixelclassifier.settings as pc_settings
from omr.stafflines.detection.dataset import PCDataset
from linesegmentation.detection.callback import LineDetectionCallback

logger = logging.getLogger(__name__)


class PCPredictionCallback(LineDetectionCallback):
    def __init__(self, callback: LineDetectionPredictorCallback):
        self.callback = callback
        super().__init__(7, 1)

    def changed(self):
        self.callback.progress_updated(self.get_current_page_progress())


class BasicStaffLinePredictor(StaffLinesPredictor):
    def __init__(self, params: StaffLinePredictorParameters):
        super().__init__()
        self.params = params
        model_path = None

        if params.checkpoints is not None and len(params.checkpoints) > 0:
            if len(params.checkpoints) > 1:
                logger.warning("Only one or no model is allowed for the basic staff line predictor, but got {}: {}".format(
                    len(params.checkpoints), params.checkpoints))

            model_path = params.checkpoints[0]
            if not os.path.exists(model_path + '.meta'):
                logger.debug('Model not found at {}. Using staff line detection without a model.'.format(model_path))
                model_path = None
            else:
                logger.info('Running line detection with model {}.'.format(model_path))

        if not model_path:
            logger.info('Running line detection without a model.')

        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            min_lines_per_system=params.min_num_staff_lines,
            line_number=params.num_staff_lines,
            horizontal_min_length=6,
            line_interpolation=True,
            line_space_height=0,
            target_line_space_height=params.target_line_space_height,
            model=model_path,
            post_process=params.post_processing,
            best_fit_scale=params.best_fit_scale,
            # debug=True, smooth_lines_advdebug=True,
        )
        self.line_detection = LineDetection(self.settings)

    def predict(self, pcgts_files: List[PcGts], callback: Optional[LineDetectionPredictorCallback] = None) -> PredictionType:
        pc_dataset = PCDataset(pcgts_files, self.params.dataset_params)
        dataset = pc_dataset.to_line_detection_dataset()
        gray_images = [(255 - data.line_image).astype(np.uint8) for data in dataset]
        if callback:
            # TODO: Line detection callback of line-detection not as class member variable
            self.line_detection.callback = PCPredictionCallback(callback)

        predictions = self.line_detection.detect(gray_images)
        for i, (data, r) in enumerate(zip(dataset, predictions)):
            rlmd: RegionLineMaskData = data
            logger.debug("Predicted {}/{}. File {}".format(i + 1, len(dataset), rlmd.operation.page.location.local_path()))
            if len(r) == 0:
                logger.warning('No staff lines detected.')
                yield PredictionResult(MusicLines(), MusicLines(), rlmd)
            else:
                def transform_points(yx_points):
                    return Coords(np.array([pc_dataset.line_and_mask_operations.local_to_global_pos(Point(p[1], p[0]), rlmd.operation.params).p for p in yx_points]))

                ml_global = MusicLines([MusicLine(staff_lines=StaffLines([StaffLine(transform_points(list(pl))) for pl in l])) for l in r])
                ml_local = MusicLines([MusicLine(staff_lines=StaffLines([StaffLine(Coords(np.array(pl)[:, ::-1])) for pl in l])) for l in r])
                yield PredictionResult(ml_global, ml_local, rlmd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    # page = Book('demo').page('page00000001')
    book = DatabaseBook('Graduel_Part_1')
    # page = book.page('Graduel_de_leglise_de_Nevers_032_rot')  # zacken in linie
    # page = book.page('Graduel_de_leglise_de_Nevers_531')
    page = book.page('Graduel_de_leglise_de_Nevers_030')
    #page = book.page('Graduel_de_leglise_de_Nevers_520')
    #page = book.page('Graduel_de_leglise_de_Nevers_513')

    pcgts = [PcGts.from_file(page.file('pcgts'))]

    params = StaffLinePredictorParameters(
        # None if False else [book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))],
        # ["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/internal_storage/default_models/french14/pc_staff_lines/model"],
        ["/home/wick/Documents/Projects/ommr4all-deploy/modules/ommr4all-server/models_out/all/line_detection_4/best"],
        # ["/home/wick/Downloads/line_detection_0/best"],
        target_line_space_height=10,
        dataset_params=StaffLineDetectionDatasetParams(
            gt_required=True,
            full_page=True,
            gray=True,
            pad=0,
            extract_region_only=False,
            gt_line_thickness=3,
        )

    )
    detector = BasicStaffLinePredictor(params)
    for prediction in detector.predict(pcgts):
        f, ax = plt.subplots(1, 3)
        staffs = prediction.music_lines_local
        data = prediction.line
        img = np.array(data.line_image, dtype=np.uint8)
        ax[0].imshow(255 - img, cmap='gray')
        staffs.draw(img, color=255, line_thickness=3)
        s = np.zeros(img.shape)
        staffs.draw(s, color=255, line_thickness=3)
        ax[1].imshow(img)
        ax[2].imshow(s, cmap='gray')
        plt.show()

