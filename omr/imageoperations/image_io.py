from database.file_formats.pcgts import Point
from . import ImageOperation, ImageOperationData, OperationOutput, ImageData
from copy import copy
from PIL import Image
import numpy as np
import PIL.ImageOps
from typing import List, Tuple, Any


class ImageLoadFromPageOperation(ImageOperation):
    def __init__(self, invert=False, files: List[Tuple[str, bool]] = None):
        super().__init__()
        self.invert = invert
        self.files: List[Tuple[str, bool]] = files if files else [('gray_norm', False)]

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        d = copy(data)

        book_page = data.page.location
        d.images = []
        for file, nn_rescale in self.files:
            img = Image.open(book_page.file(file, create_if_not_existing=True).local_path())
            if self.invert:
                img = PIL.ImageOps.invert(img)

            d.images.append(ImageData(np.array(img), nn_rescale))
        d.params = None
        return [d]

    def local_to_global_pos(self, p, params):
        return p

    def global_to_local_pos(self, p: Point, params: Any) -> Point:
        return p
