from abc import ABC, abstractmethod
from PIL import Image
import numpy as np

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
        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        self.binarizer = binarizer

        self.angle = 0
        self.original_out = None
        self.gray_out = None
        self.binary_out = None

    def deskew(self,
               original_image: Image,
               gray_image: Image,
               binary_image: Image,
               ):
        self.original_image = original_image
        self.gray_image = gray_image
        self.binary_image = binary_image
        self.angle = self._estimate_skew_angle()
        self.original_out = self.original_image.rotate(self.angle)
        self.gray_out = self.gray_image.rotate(self.angle)
        self.binary_out = self.binarizer.binarize(self.original_out)
        return self.original_out, self.gray_out, self.binary_out

    @abstractmethod
    def _estimate_skew_angle(self) -> float:
        return 0

    def plot(self):
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 3, sharex='all', sharey='all')
        ax[0, 0].imshow(np.array(self.original_image))
        ax[0, 1].imshow(np.array(self.gray_image))
        ax[0, 2].imshow(np.array(self.binary_image))
        ax[1, 0].imshow(np.array(self.original_out))
        ax[1, 1].imshow(np.array(self.gray_out))
        ax[1, 2].imshow(np.array(self.binary_out))
        plt.show()
