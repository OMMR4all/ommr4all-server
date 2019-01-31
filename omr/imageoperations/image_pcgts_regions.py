from . import ImageOperation, ImageOperationData, OperationOutput, ImageData
from copy import copy
from PIL import Image
from omr.datatypes.page.textregion import TextRegionType
import numpy as np
import PIL.ImageOps
from typing import List


class ImageDrawRegions(ImageOperation):
    def __init__(self, music_region=False, text_region_types=List[TextRegionType], color=0):
        super().__init__()
        self.music_region = music_region
        self.text_region_types = text_region_types
        self.color = color

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        d = copy(data)
        d.params = None

        page = data.page
        for image_data in d.images:
            image = image_data.image
            if self.music_region:
                for mr in page.music_regions:
                    mr.coords.draw(image, self.color, fill=True)
                    for ml in mr.staffs:
                        ml.coords.draw(image, self.color, fill=True)

            for r in [r for r in page.text_regions if r.region_type in self.text_region_types]:
                r.coords.draw(image, self.color, fill=True)
                for l in r.text_lines:
                    l.coords.draw(image, self.color, fill=True)

        return [d]

    def local_to_global_pos(self, p, params):
        return p
