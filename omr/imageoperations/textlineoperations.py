from database.file_formats.pcgts import Coords
from omr.imageoperations.image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from omr.imageoperations.image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any, Optional, Set
from database.file_formats.pcgts.page import BlockType, Block, Line
import numpy as np
from PIL import Image
from copy import copy
from enum import IntEnum
from omr.dewarping.dummy_dewarper import Dewarper
import logging

logger = logging.getLogger(__name__)


class ImageExtractDeskewedLyrics(ImageOperation):
    def __init__(self):
        super().__init__()

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        all_mls = data.page.all_music_lines()
        all_tls = data.page.all_text_lines()

        def extract_transformed_coords(ml: Line) -> List[Coords]:
            lines = ml.staff_lines.sorted()
            return [data.page.page_to_image_scale(sl.coords, data.scale_reference) for sl in lines]

        def transform_point(p) -> np.array:
            return data.page.page_to_image_scale(p, data.scale_reference)

        s = [extract_transformed_coords(ml) for ml in all_mls]

        # dewarp
        images = [Image.fromarray(image)]
        dewarper = Dewarper(images[0].size, s)
        dew_page, = tuple(map(np.array, dewarper.dewarp(images)))
        out = []

        for tl in all_tls:
            dew_coords = Coords(dewarper.transform_points(transform_point(tl.coords.points)))
            aabb = dew_coords.aabb()
            if aabb.area() == 0:
                continue

            image = dew_page[int(aabb.top()):int(aabb.bottom()), int(aabb.left()):int(aabb.right())]

            img_data = copy(data)
            img_data.page_image = image
            img_data.text_line = tl
            img_data.images = [ImageData(image, True)]
            img_data.params = (dewarper, aabb)
            out.append(img_data)

        if False:
            import matplotlib.pyplot as plt
            if True:
                f, ax = plt.subplots(len(out), 1)
                for o, a in zip(out, ax):
                    a.imshow(o.images[0].image)

                plt.show()
            else:
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(image)
                ax[1].imshow(dew_page)
                plt.show()

        return out

    def local_to_global_pos(self, p: Point, params: Any):
        (dewarper, aabb) = params
        return Point(dewarper.transform_point(p.p + aabb.tl.p))


# extract image of a text from the binary image
class ImageExtractTextLineImages(ImageOperation):
    def __init__(self, text_region_types: Set[BlockType], pad=0, extract_region_only=True):
        super().__init__()
        self.text_region_types = text_region_types
        self.pad = pad
        self.extract_region_only = extract_region_only
        self.cropper = ImageCropToSmallestBoxOperation(pad=pad)

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        marked_regions = np.zeros(image.shape, dtype=np.uint8)
        i = 1
        text_blocks = [tr for tr in data.page.blocks_of_type(self.text_region_types)]

        def p2i(p):
            return data.page.page_to_image_scale(p, data.scale_reference)

        for tr in text_blocks:
            for tl in tr.lines:
                p2i(tl.coords).draw(marked_regions, i, 0, fill=True)
                i += 1

        out = []

        i = 1
        for tr in text_blocks:
            for tl in tr.lines:
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
