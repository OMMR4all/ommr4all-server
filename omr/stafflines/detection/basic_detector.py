from omr.stafflines.detection.staffline_detector import StaffLineDetector
import main.book as book
from omr.datatypes import MusicLines, MusicLine, StaffLine, StaffLines, Coords
import numpy as np
import os


class BasicStaffLineDetector(StaffLineDetector):
    def __init__(self, page: book.Page):
        super().__init__()
        model_path = page.book.local_path(os.path.join('staff_lines', 'model'))
        if not os.path.exists(model_path + '.meta'):
            model_path = None

        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            numLine=4,
            minLength=6,
            lineExtension=True,
            debug=False,
            lineSpaceHeight=0,
            targetLineSpaceHeight=10,
            model=model_path
        )
        self.line_detection = LineDetection(self.settings)

    def detect(self, binary_path: str, gray_path: str) -> MusicLines:
        r = list(self.line_detection.detect([gray_path]))[0]
        ml = MusicLines([MusicLine(staff_lines=StaffLines([StaffLine(Coords(np.round(np.asarray(pl)[:, ::-1]))) for pl in l])) for l in r])
        ml.approximate_staff_lines()
        from PIL import Image
        g = np.array(Image.open(gray_path)) / 255
        ml.fit_staff_lines_to_gray_image(g)
        return ml


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    from main.book import Book
    page = Book('demo').page('page00000001')
    binary = page.file('binary_deskewed').local_path()
    gray = page.file('gray_deskewed').local_path()

    detector = BasicStaffLineDetector(page)
    staffs = detector.detect(binary, gray)
    img = np.array(Image.open(page.file('color_deskewed').local_path()), dtype=np.uint8)
    staffs.draw(img)
    plt.imshow(img)
    plt.show()

