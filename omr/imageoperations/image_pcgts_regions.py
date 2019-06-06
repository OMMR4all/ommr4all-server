from . import ImageOperation, ImageOperationData, OperationOutput
from copy import copy
from database.file_formats.pcgts.page.textregion import TextRegionType
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

        if self.music_region or len(self.text_region_types) > 0:
            page = data.page
            for image_data in d.images:
                image = image_data.image
                if self.music_region:
                    for mr in page.music_regions:
                        page.page_to_image_scale(mr.coords, data.scale_reference).draw(image, self.color, fill=True)
                        for ml in mr.staffs:
                            page.page_to_image_scale(ml.coords, data.scale_reference).draw(image, self.color, fill=True)

                for r in [r for r in page.text_regions if r.region_type in self.text_region_types]:
                    page.page_to_image_scale(r.coords, data.scale_reference).draw(image, self.color, fill=True)
                    for l in r.text_lines:
                        page.page_to_image_scale(l.coords, data.scale_reference).draw(image, self.color, fill=True)

        return [d]

    def local_to_global_pos(self, p, params):
        return p
