import numpy as np
from skimage.measure import approximate_polygon
import cv2


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return self.to_string()

    @staticmethod
    def from_string(s):
        return Point(*tuple(map(float, s.split(","))))

    def to_string(self):
        return ",".join(map(str, [self.x, self.y]))

    @staticmethod
    def from_json(json):
        return Point.from_string(json)

    def to_json(self):
        return self.to_string()


class Coords:
    def __init__(self, points: np.ndarray = np.zeros((0, 2), dtype=float)):
        self.points = np.array(points, dtype=float)

    def __str__(self):
        return self.to_string()

    @staticmethod
    def from_string(s):
        if len(s) == 0:
            return Coords()

        return Coords(np.array([list(map(float, p.split(','))) for p in s.split(" ")]))

    def to_string(self):
        return " ".join(",".join(map(str, self.points[i])) for i in range(self.points.shape[0]))

    @staticmethod
    def from_json(json):
        return Coords.from_string(json)

    def to_json(self):
        return self.to_string()

    def interpolate_y(self, x):
        return np.interp(x, self.points[:, 0], self.points[:, 1])

    def approximate(self, distance):
        self.points = approximate_polygon(self.points, distance)

    def draw(self, canvas, color=(0, 255, 0), thickness=5):
        pts = self.points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, int(thickness))


if __name__ == '__main__':
    c = Coords(np.array([[0, 1], [1, 2], [6, -123]]))
    print(c.to_json())
    print(Coords.from_json(c.to_json()).to_json() == c.to_json())

    p = Point(-20, 100)
    print(p.to_json())
    print(Point.from_json(p.to_json()).to_json() == p.to_json())
