from abc import ABC, abstractmethod


def default_binarizer():
    from .ocropus_binarizer import OCRopusBin
    return OCRopusBin()


class Binarizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def binarize(self, image):
        return image



