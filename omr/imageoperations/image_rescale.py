from .image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point
from typing import Tuple, List, NamedTuple, Any
import numpy as np
from scipy.ndimage import interpolation
from scipy.misc import imresize


class ImageScaleOperation(ImageOperation):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        s = [imresize(d.image, self.factor, interp='nearest' if d.nearest_neighbour_rescale else 'bilinear') for d in data]
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

    @staticmethod
    def scale_to_h(img, target_height, order=1, cval=0):
        h, w = img.shape
        if h == 0:
            return np.zeros((0, target_height))

        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        output = interpolation.affine_transform(
            img,
            np.eye(2) / scale,
            order=order,
            output_shape=(target_height,target_width),
            mode='constant',
            cval=cval)

        output = np.array(output, dtype=img.dtype)
        return output
