from omr.imageoperations.image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from omr.imageoperations.image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any, Optional, Set
from database.file_formats.pcgts import TextLine, TextRegion, TextRegionType
import numpy as np
from PIL import Image
from copy import copy
from enum import IntEnum
from omr.dewarping.dummy_dewarper import dewarp, transform
import logging

logger = logging.getLogger(__name__)


# extract image of a text from the binary image
class ImageExtractTextLineImages(ImageOperation):
    def __init__(self, text_region_types: Set[TextRegionType], pad=0, extract_region_only=True):
        super().__init__()
        self.text_region_types = text_region_types
        self.pad = pad
        self.extract_region_only = extract_region_only
        self.cropper = ImageCropToSmallestBoxOperation(pad=pad)

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        marked_regions = np.zeros(image.shape, dtype=np.uint8)
        i = 1
        text_regions = [tr for tr in data.page.text_regions if tr.region_type in self.text_region_types]
        for tr in text_regions:
            for tl in tr.text_lines:
                tl.coords.draw(marked_regions, i, 0, fill=True)
                i += 1

        out = []

        i = 1
        for tr in text_regions:
            for tl in tr.text_lines:
                mask = marked_regions == i
                if len(tl.text()) == 0:
                    continue

                if np.sum(mask) == 0:  # empty mask, skip
                    continue
                else:
                    img_data = copy(data)
                    img_data.page_image = image
                    img_data.text_line = tl
                    img_data.images = [ImageData(mask, True), ImageData(image, True)]
                    cropped = self.cropper.apply_single(img_data)[0]
                    self._extract_image_op(img_data)

                    img_data.params = (i, cropped.params)
                    out.append(img_data)

                i += 1

        return out

    def _extract_image_op(self, data: ImageOperationData):
        data.images = [
                          data.images[1],
                          ImageData(data.images[0].image * data.images[1].image, False) if self.extract_region_only else data.images[1]
                      ]

    def local_to_global_pos(self, p: Point, params: Any):
        i, (t, b, l, r) = params
        # default operations
        return Point(p.x + l, t + p.y)
