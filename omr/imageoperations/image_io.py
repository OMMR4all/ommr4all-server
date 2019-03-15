from . import ImageOperation, ImageOperationData, OperationOutput, ImageData
from copy import copy
from PIL import Image
import numpy as np
import PIL.ImageOps
from typing import List


class ImageLoadFromPageOperation(ImageOperation):
    def __init__(self, invert=False, files: List[str] = None):
        super().__init__()
        self.invert = invert
        self.files = files if files else ['gray_deskewed']

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        d = copy(data)

        book_page = data.page.location
        d.images = []
        for file in self.files:
            img = Image.open(book_page.file(file).local_path())
            if self.invert:
                img = PIL.ImageOps.invert(img)

            d.images.append(ImageData(np.array(img), False))
        d.params = None
        return [d]

    def local_to_global_pos(self, p, params):
        return p
