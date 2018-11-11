from .image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point
from typing import Tuple, List, NamedTuple, Any
import numpy as np
from scipy.ndimage import interpolation


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
        scale = target_height * 1.0 / h
        target_width = np.maximum(int(scale * w), 1)
        output = interpolation.affine_transform(
            1.0 * img,
            np.eye(2) / scale,
            order=order,
            output_shape=(target_height,target_width),
            mode='constant',
            cval=cval)

        output = np.array(output, dtype=img.dtype)
        return output
