from .image_operation import ImageOperation, ImageDataInput, OperationOutput, ImageData, Point
from typing import Tuple, List, NamedTuple, Any
import numpy as np


class Rect(NamedTuple):
    t: int
    b: int
    l: int
    r: int


class ImageCropToSmallestBoxOperation(ImageOperation):
    def __init__(self):
        super().__init__()

    def apply_single(self, data: ImageDataInput) -> OperationOutput:
        def smallestbox(a, datas) -> Tuple[List[np.ndarray], Rect]:
            r = a.any(1)
            m, n = a.shape
            c = a.any(0)
            q, w, e, r = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
            return [d[q:w, e:r] for d in datas], Rect(q, w, e, r)

        imgs, r = smallestbox([d.image for d in data])
        data.images = [ImageData(i, d.nearest_neighbour_rescale) for d, i in zip(data, imgs)]
        return OperationOutput(data, r)

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        r: Rect = params
        return Point(p.x + r.l, p.y + p.t)


class ImagePadToPowerOf2(ImageOperation):
    def __init__(self, power=3):
        super().__init__()
        self.power = power

    def apply_single(self, data: ImageDataInput) -> OperationOutput:
        x, y = data.images[0].image.shape

        f = 2 ** 3
        tx = (((x // 2) // 2) // 2) * 8
        ty = (((y // 2) // 2) // 2) * 8

        if x % f != 0:
            px = tx - x + f
            x = x + px
        else:
            px = 0

        if y % f != 0:
            py = ty - y + f
            y = y + py
        else:
            py = 0

        pad = ((px, 0), (py, 0))

        data.images = [ImageData(np.pad(d.image, pad, 'edge'), d.nearest_neighbour_rescale) for d in data]
        return OperationOutput(data, (px, py))

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        px, py = params
        return Point(p.x + px, py.y + py)
