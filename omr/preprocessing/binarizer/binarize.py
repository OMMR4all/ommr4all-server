from abc import ABC, abstractmethod

class Binarizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def binarize(self, image):
        return image



