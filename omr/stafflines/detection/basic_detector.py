from omr.stafflines.detection.staffline_detector import StaffLineDetector
from omr.datatypes import MusicLines, MusicLine, StaffLine, StaffLines, Coords
import numpy as np


class BasicStaffLineDetector(StaffLineDetector):
    def __init__(self):
        super().__init__()

        from linesegmentation.detection import LineDetectionSettings, LineDetection
        self.settings = LineDetectionSettings(
            numLine=4,
            minLength=6,
            lineExtension=True,
            debug=False,
            lineSpaceHeight=0,
            targetLineSpaceHeight=10,
            model=None,
        )
        self.line_detection = LineDetection(self.settings)

    def detect(self, binary_path: str, gray_path: str) -> MusicLines:
        r = list(self.line_detection.detectbasic([gray_path]))[0]
        ml = MusicLines([MusicLine(staff_lines=StaffLines([StaffLine(Coords(np.asarray(pl)[:, ::-1])) for pl in l])) for l in r])
        ml.approximate_staff_lines()
        return ml


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    from main.book import Book
    page = Book('demo').page('page00000001')
    binary = page.file('binary_deskewed').local_path()
    gray = page.file('gray_deskewed').local_path()

    detector = BasicStaffLineDetector()
    staffs = detector.detect(binary, gray)
    img = np.array(Image.open(page.file('color_deskewed').local_path()), dtype=np.uint8)
    staffs.draw(img)
    plt.imshow(img)
    plt.show()

