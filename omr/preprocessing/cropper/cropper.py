from abc import ABC, abstractmethod
from collections import namedtuple
from PIL import Image


def default_cropper():
    from .noop_cropper import NoopCropper
    return NoopCropper()


class Cropper(ABC):
    Bounds = namedtuple('Rect', ['t', 'l', 'r', 'b'])

    def __init__(self):
        super().__init__()

        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        self.rect = Cropper.Bounds(0, 0, -1, -1)
        self.original_out = None
        self.gray_out = None
        self.binary_out = None

    def crop(self, original_image: Image, gray_image: Image, binary_image: Image):
        self.original_image = original_image
        self.gray_image = gray_image
        self.binary_image = binary_image

        self.rect = self._content_rect()

        images = [self.original_image, self.gray_image, self.binary_image]
        self.original_out, self.gray_out, self.binary_out = tuple([i.crop(self.rect)for i in images])
        return self.original_out, self.gray_out, self.binary_out

    @abstractmethod
    def _content_rect(self) -> Bounds:
        w, h = self.original_image.size
        return 0, 0, w, h


