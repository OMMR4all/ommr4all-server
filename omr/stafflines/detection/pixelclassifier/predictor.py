from omr.stafflines.detection.predictor import StaffLinesPredictor
from database import DatabasePage, DatabaseBook
from database.file_formats.pcgts import *
import numpy as np
import os
import logging
import omr.stafflines.detection.pixelclassifier.settings as pc_settings

logger = logging.getLogger(__name__)


class BasicStaffLinePredictor(StaffLinesPredictor):
    def __init__(self, page: DatabasePage):
        super().__init__()
        # used specific model or use default as fallback if existing
        model_path = page.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name))
        if not os.path.exists(model_path + '.meta'):
            model_path_new = os.path.join(page.book.local_default_models_path(os.path.join(pc_settings.model_dir, pc_settings.model_name)))
            logger.debug('Book specific model not found at {}. Trying general model at {}'.format(
                model_path, model_path_new
            ))
            model_path = model_path_new

        if not os.path.exists(model_path + '.meta'):
            logger.debug('Global model not found at {}. Using staff line detection without a model.'.format(model_path))
            model_path = None
            logger.info('Running line detection without a model.')
        else:
            logger.info('Running line detection with model {}.'.format(model_path))

        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            numLine=4,
            minLength=6,
            lineExtension=True,
            debug=False,
            lineSpaceHeight=20,
            targetLineSpaceHeight=10,
            model=model_path,
            post_process=True,
            smooth_lines=2,
            line_fit_distance=1,
        )
        self.line_detection = LineDetection(self.settings)

    def detect(self, binary_path: str, gray_path: str) -> MusicLines:
        r = list(self.line_detection.detect([gray_path]))[0]
        if len(r) == 0:
            logger.warning('No staff lines detected.')
            return MusicLines()

        ml = MusicLines([MusicLine(staff_lines=StaffLines([StaffLine(Coords((np.asarray(list(pl))[:, ::-1]))) for pl in l])) for l in r])
        # ml.approximate_staff_lines()
        # from PIL import Image
        # g = np.array(Image.open(gray_path)) / 255
        # ml.fit_staff_lines_to_gray_image(g)
        return ml


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    # page = Book('demo').page('page00000001')
    page = DatabaseBook('Graduel').pages()[0]
    binary = page.file('binary_deskewed').local_path()
    gray = page.file('gray_deskewed').local_path()

    detector = BasicStaffLinePredictor(page)
    staffs = detector.detect(binary, gray)
    img = np.array(Image.open(page.file('color_deskewed').local_path()), dtype=np.uint8)
    staffs.draw(img)
    plt.imshow(img)
    plt.show()

