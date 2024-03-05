from .image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point
from typing import Tuple, List, NamedTuple, Any
import numpy as np
from skimage.transform import resize


class ImageScaleOperation(ImageOperation):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        s = [resize(d.image, self.factor, order=0 if d.nearest_neighbour_rescale else 1) for d in data]
        scale = s[0].shape[1] / data.images[0].image.shape[1]
        data.images = [ImageData(i, d.nearest_neighbour_rescale) for d, i in zip(data, s)]
        data.params = (scale, )
        return [data]

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        return Point(p.x / self.factor, p.y / self.factor)


class ImageRescaleToHeightOperation(ImageOperation):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        s = [ImageRescaleToHeightOperation.scale_to_h(d.image, self.height, 0 if d.nearest_neighbour_rescale else 3) for d in data]
        scale = s[0].shape[1] / data.images[0].image.shape[1]
        data.images = [ImageData(i, d.nearest_neighbour_rescale) for d, i in zip(data, s)]
        data.params = (scale, )
        return [data]

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        scale, = params
        return Point(p.x / scale, p.y / scale)

    def global_to_local_pos(self, p: Point, params: Any) -> Point:
        scale, = params
        return Point(p.x * scale, p.y * scale)
    @staticmethod
    def scale_to_h(img, target_height, order=1, cval=0):
        assert(img.dtype == np.uint8)
        h, w = img.shape[0:2]
        if h == 0:
            return np.zeros((0, target_height))

        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        output = resize(img, (target_height, target_width),
                        order=0, mode='edge', cval=cval, preserve_range=True,
                        anti_aliasing=order).astype(np.uint8)
        return output
