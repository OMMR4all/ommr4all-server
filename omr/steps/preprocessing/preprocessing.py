from PIL import Image
from omr.steps.preprocessing.binarizer import default_binarizer
from omr.steps.preprocessing.gray.img2gray import im2gray
from omr.steps.preprocessing.deskewer import default_deskewer


class Preprocessing:
    def __init__(self, operation_width=500):
        self.operation_width = operation_width
        self.binarizer = default_binarizer()
        self.deskewer = default_deskewer(self.binarizer)

    def binarize(self, color: Image):
        return self.binarizer.binarize(color)

    def im2gray(self, color: Image):
        return im2gray(color)

    def preprocess(self, original: Image):
        o_w, o_h = original.size
        scale = self.operation_width / o_w
        if self.operation_width < o_w:
            working_image = original.resize((int(scale * o_w), int(scale * o_h)), Image.BILINEAR)
        else:
            working_image = original

        working_bin = self.binarize(working_image)
        working_gray = self.im2gray(working_image)

        angle = self.deskewer.estimate_skew_angle(working_image, working_gray, working_bin)
        working_image = original.rotate(angle)
        working_gray = self.im2gray(working_image)
        working_bin = self.binarizer.binarize(working_image)

        return working_image, working_gray, working_bin




