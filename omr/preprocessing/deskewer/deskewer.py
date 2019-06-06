from abc import ABC, abstractmethod
from PIL import Image

from ..binarizer import default_binarizer

def default_deskewer(
        binarizer=default_binarizer()
        ):
    from . import OcropusDeskewer
    return OcropusDeskewer(binarizer)


class Deskewer(ABC):
    def __init__(self,
                 binarizer
                 ):
        self.binarizer = binarizer

    def estimate_skew_angle(self,
                            original_image: Image,
                            gray_image: Image,
                            binary_image: Image,
                            ):
        return self._estimate_skew_angle(original_image, gray_image, binary_image)

    @abstractmethod
    def _estimate_skew_angle(self, color_image, gray_image, binary_image):
        return 0
