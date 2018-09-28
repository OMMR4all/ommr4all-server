import numpy as np
from typing import List, NamedTuple
import cv2
from .json_util import line_to_json

class TextBoundaries():
    def __init__(self, capital=list(), upper=list(), lower=list(), below=list()):
        self.capital = np.array(capital, np.int32)
        self.upper = np.array(upper, np.int32)
        self.lower = np.array(lower, np.int32)
        self.below = np.array(below, np.int32)

    def lines(self):
        return [self.capital, self.upper, self.lower, self.below]

    def to_json(self):
        return {
            'capital': {'points': line_to_json(self.capital)},
            'upper': {'points': line_to_json(self.upper)},
            'lower': {'points': line_to_json(self.lower)},
            'below': {'points': line_to_json(self.below)},
        }


class TextLine:
    def __init__(self, polygons: List[np.ndarray] = list(), boundaries: TextBoundaries = TextBoundaries()):
        self.polygons = [np.array(p, np.int32) for p in polygons]
        self.boundaries = boundaries

    def draw(self, image):
        cv2.polylines(image, self.polygons, True, (0, 255, 0), 5)
        cv2.polylines(image, self.boundaries.lines(), False, (0, 0, 255), 2)

    def to_json(self):
        return {
            'polygons': [{'points': line_to_json(poly)} for poly in self.polygons],
            'boundaries': self.boundaries.to_json(),
        }
