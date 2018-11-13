from . import ImageOperation, ImageOperationData, OperationOutput, ImageData
from copy import copy
from PIL import Image
import numpy as np
import PIL.ImageOps


class ImageLoadFromPageOperation(ImageOperation):
    def __init__(self, invert=False):
        super().__init__()
        self.invert = invert

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        d = copy(data)

        book_page = data.page.location
        bin = Image.open(book_page.file('gray_deskewed').local_path())
        if self.invert:
            bin = PIL.ImageOps.invert(bin)

        d.images = [ImageData(np.array(bin), False)]
        d.params = None
        return [d]

    def local_to_global_pos(self, p, params):
        return p
