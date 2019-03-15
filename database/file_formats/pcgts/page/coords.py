import numpy as np
from skimage.measure import approximate_polygon
import cv2
from typing import Type, Union


class Point:
    def __init__(self, x: Union[int, float, np.ndarray, 'Size', 'Point'] = 0, y=0):
        if isinstance(x, np.ndarray):
            self.p = x
        elif isinstance(x, Size) or isinstance(x, Point):
            self.p = x.p
        else:
            self.p = np.array([x, y])

    @property
    def x(self):
        return self.p[0]

    @property
    def y(self):
        return self.p[1]

    def distance_sqr(self, p: 'Point') -> float:
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2

    def astype(self, dtype):
        return Point(self.p.astype(dtype))

    def round(self, decimals=0, out=None):
        return Point(np.round(self, decimals, out))

    def xy(self):
        return self.x, self.y

    def yx(self):
        return self.y, self.x

    def __str__(self):
        return self.to_string()

    def __add__(self, p):
        if isinstance(p, Point):
            return Size(self.p + p.p)
        elif isinstance(p, Size):
            return Point(self.p + p.p)
        else:
            return Point(self.p + p)

    def __sub__(self, p):
        if isinstance(p, Point):
            return Size(self.p - p.p)
        elif isinstance(p, Size):
            return Point(self.p - p.p)
        else:
            return Point(self.p - p)

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


class Size:
    def __init__(self, w: Union[int, float, np.ndarray, 'Size', 'Point'] = 0, h=0):
        if isinstance(w, np.ndarray):
            self.p = w
        elif isinstance(w, Size) or isinstance(w, Point):
            self.p = w.p
        else:
            self.p = np.array([w, h])

    @property
    def w(self):
        return self.p[0]

    @property
    def h(self):
        return self.p[1]

    def astype(self, dtype):
        return Size(self.p.astype(dtype))

    def round(self, decimals=0, out=None):
        return Size(np.round(self, decimals, out))

    def wh(self):
        return self.w, self.h

    def hw(self):
        return self.h, self.w

    def __str__(self):
        return self.to_string()

    @staticmethod
    def from_string(s):
        return Size(*tuple(map(float, s.split(","))))

    def to_string(self):
        return ",".join(map(str, [self.w, self.h]))

    @staticmethod
    def from_json(json):
        return Size.from_string(json)

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

    def draw(self, canvas, color=(0, 255, 0), thickness=5, fill=False, offset=(0, 0)):
        pts = np.round((self.points + offset).reshape((-1, 1, 2))).astype(np.int32)
        if thickness > 0 and len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, int(thickness))
        if fill and len(pts) >= 3:
            cv2.fillPoly(canvas, [pts], color)

    def aabb(self):
        if len(self.points) == 0:
            return Rect()

        tl = self.points[0]
        br = self.points[0]
        for p in self.points[1:]:
            tl = np.min([tl, p], axis=0)
            br = np.max([br, p], axis=0)

        Point(tl)

        return Rect(Point(tl), Point(br))

    def extract_from_image(self, image: np.ndarray):
        aabb = self.aabb()
        sub_image = image[int(aabb.tl[1]):int(aabb.br[1]), int(aabb.tl[0]):int(aabb.br[0])]
        return sub_image


class Rect:
    def __init__(self, origin: Point = None, size: Union[Point, Size] = None):
        self.origin = origin if origin else Point()
        if size and isinstance(size, Point):
            self.size = Size(size - origin)
        else:
            self.size = size if size else Size()

    @property
    def tl(self):
        return self.origin

    @property
    def br(self):
        return self.origin + self.size

    @staticmethod
    def from_json(json: dict):
        if json is None:
            return Rect()

        return Rect(
            Point.from_json(json.get("origin", "0,0")),
            Size.from_json(json.get("size", "0,0")),
        )

    def to_json(self):
        return {
            "origin": self.origin.to_json(),
            "size": self.size.to_json(),
        }


if __name__ == '__main__':
    c = Coords(np.array([[0, 1], [1, 2], [6, -123]]))
    print(c.to_json())
    print(Coords.from_json(c.to_json()).to_json() == c.to_json())
    print(c.aabb())

    p = Point(-20, 100).astype(float)
    print(p.to_json())
    print(Point.from_json(p.to_json()).to_json() == p.to_json())
