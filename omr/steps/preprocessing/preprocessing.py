from PIL import Image
from omr.steps.preprocessing.binarizer import default_binarizer
from omr.steps.preprocessing.gray.img2gray import im2gray
from omr.steps.preprocessing.deskewer import default_deskewer


class Preprocessing:
    def __init__(self, operation_max_width=1000, deskew=True):
        self.operation_max_width = operation_max_width
        self.binarizer = default_binarizer()
        self.deskewer = default_deskewer(self.binarizer)
        self.deskewed_angle = -1
        self.deskew = deskew

    def binarize(self, color: Image):
        return self.binarizer.binarize(color)

    def im2gray(self, color: Image):
        return im2gray(color)

    def preprocess(self, original: Image):
        if self.deskew:
            o_w, o_h = original.size
            scale = self.operation_max_width / o_w
            if self.operation_max_width < o_w:
                working_image = original.resize((int(scale * o_w), int(scale * o_h)), Image.BILINEAR)
            else:
                working_image = original

            working_bin = self.binarize(working_image)
            working_gray = self.im2gray(working_image)
            self.deskewed_angle = self.deskewer.estimate_skew_angle(working_image, working_gray, working_bin)
            working_image = original.rotate(self.deskewed_angle)
        else:
            working_image = original
        working_gray = self.im2gray(working_image)
        working_bin = self.binarizer.binarize(working_image)

        return working_image, working_gray, working_bin


if __name__ == '__main__':
    from omr.steps.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import os
    original = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'pages', 'page00000002', 'color_original.jpg'))
    preproc = Preprocessing()
    preproc.preprocess(original)
