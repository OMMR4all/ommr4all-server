from .image_operation import ImageOperation, ImageDataInput, OperationOutput, ImageData, Point
from .image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any
from omr.datatypes import Page
import numpy as np
import PIL.ImageOps
from PIL import Image


class ImageExtractDewarpedStaffLineImages(ImageOperation):
    def __init__(self, data_image: np.ndarray, page: Page):
        super().__init__()
        labels = np.zeros(data_image.size[::-1], dtype=np.uint8)
        i = 1
        s = []
        for mr in page.music_regions:
            for ml in mr.staffs:
                s.append(ml)
                ml.coords.draw(labels, i, 0, fill=True)
                i += 1

        from omr.dewarping.dummy_dewarper import dewarp
        dew_page, dew_labels = tuple(map(np.array, dewarp([bin, Image.fromarray(labels)], s, None)))

        i = 1
        for mr in page.music_regions:
            for ml in mr.staffs:
                mask = dew_labels == i
                data = ImageDataInput([ImageData(mask, True), ImageData(dew_page, False)])
                out = self.cropper.apply_single(data)

                out, rect = smallestbox(mask, [mask, dew_page])
                mask, line = tuple(out)
                # yield ml, line * np.stack([mask] * 3, axis=-1), str
                yield LoadedImage(ml, line * mask, bin, rect, "")
                i += 1

    def local_to_global_pos(self, p: Point, params: Any):
        pass

class ImageExtractMusicLineOperation(ImageOperation):
    def __init__(self):
        super().__init__()

    def apply_single(self, data: ImageDataInput) -> OperationOutput:
        pass
