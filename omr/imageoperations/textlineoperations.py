from database.file_formats.pcgts import Coords
from omr.imageoperations.image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from omr.imageoperations.image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any, Optional, Set
from database.file_formats.pcgts.page import BlockType, Block, Line
import numpy as np
from PIL import Image
from copy import copy
from enum import IntEnum
from omr.dewarping.dummy_dewarper import dewarp, transform
import logging

logger = logging.getLogger(__name__)


class ImageExtractLyricsBasedOnStaffLines(ImageOperation):
    def __init__(self):
        super().__init__()

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        s: List[List[Coords]] = []

        def extract_transformed_coords(ml: Line) -> List[Coords]:
            lines = ml.staff_lines.sorted()
            return [data.page.page_to_image_scale(sl.coords, data.scale_reference) for sl in lines]

        def transform_point(p) -> np.array:
            return data.page.page_to_image_scale(p, data.scale_reference)

        for mr in data.page.music_blocks():
            for ml in mr.lines:
                coords = extract_transformed_coords(ml)
                s.append(coords)

        # dewarp
        dew_page, = tuple(map(np.array, dewarp([Image.fromarray(image)], s, None)))

        all_mls = data.page.all_music_lines()
        all_mls.sort(key=lambda ml: ml.aabb.top())

        def get_ml_below(y, ml_skip) -> Optional[Line]:
            best = None
            best_d = 1000000
            for ml in all_mls:
                if ml == ml_skip:
                    continue

                top = ml.aabb.top()
                if top < y:
                    continue

                if not best or best_d > abs(top - y):
                    best_d = abs(top - y)
                    best = ml

            return best

        # extract
        out = []
        line_images: List[np.array] = []
        avg_line_d = data.page.avg_staff_line_distance()
        avg_d = 0
        for ml in all_mls:
            try:
                ml_below = get_ml_below(ml.aabb.bottom(), ml)
                sl_top = ml.staff_lines.sorted()[-1]
                top = sl_top.dewarped_y() + avg_line_d / 5
                first_top = transform_point((sl_top.coords.points[0][0], sl_top.dewarped_y()))
                last_top = transform_point(np.array((sl_top.coords.points[-1][0], sl_top.dewarped_y())))

                if ml_below:
                    sl_bot = ml_below.staff_lines.sorted()[0]
                    avg_d = (avg_d * len(line_images) + (sl_bot.dewarped_y() - sl_top.dewarped_y())) / (len(line_images) + 1)
                    bot = sl_bot.dewarped_y() - avg_line_d / 2
                    first_bot = transform_point((sl_bot.coords.points[0][0], bot))
                    last_bot = transform_point((sl_bot.coords.points[-1][0], bot))
                else:
                    bot = top + avg_d - avg_line_d / 2
                    first_bot = transform_point((sl_top.coords.points[0][0], bot))
                    last_bot = transform_point((sl_top.coords.points[-1][0], bot))

                tl = int(min(first_top[0], first_bot[0])), int(first_top[1])
                br = int(max(last_top[0], last_bot[0])), int(last_bot[1])
                extracted_line_img = dew_page[tl[1]:br[1], tl[0]: br[0]]
                line_images.append(dew_page[tl[1]:br[1], tl[0]: br[0]])

                img_data = copy(data)
                img_data.page_image = image
                # img_data.music_region = mr
                img_data.music_line = ml
                img_data.images = [ImageData(extracted_line_img, True)]
                img_data.params = (tl, br)
                out.append(img_data)
            except IndexError:
                continue

        if False:
            import matplotlib.pyplot as plt
            if True:
                f, ax = plt.subplots(len(line_images), 1)
                for i, a in zip(line_images, ax):
                    a.imshow(i)

                plt.show()
            else:
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(image)
                ax[1].imshow(dew_page)
                plt.show()

        return out

    def local_to_global_pos(self, p: Point, params: Any):
        ((l, t), (r, b)) = params
        # default operations
        return Point(p.x + l, t + p.y)


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
