import numpy as np
from skimage.measure import approximate_polygon
import cv2
from typing import Type, Union
from mashumaro.types import SerializableType


class Point:
    def __init__(self, x: Union[int, float, np.ndarray, 'Size', 'Point'] = 0, y=0):
        if isinstance(x, np.ndarray):
            self.p = x
        elif isinstance(x, Size) or isinstance(x, Point):
            self.p = x.p
        else:
            self.p = np.array([x, y])

    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def rotate(self, degree, origin):
        def rotate_point(xy, radians, origin=(0, 0)):
            x, y = xy
            offset_x, offset_y = origin
            adjusted_x, adjusted_y = (x - offset_x), (y - offset_y)
            cos_rad, sin_rad = np.cos(radians), np.sin(radians)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

            return qx, qy

        self.p = np.array(rotate_point(self.p, degree / 180 * np.pi, origin))

    @property
    def x(self):
        return self.p[0]

    @property
    def y(self):
        return self.p[1]

    def scale(self, scale):
        return Point(self.p * scale)

    def distance_sqr(self, p: 'Point') -> float:
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2

    def astype(self, dtype):
        return Point(self.p.astype(dtype))

    def round(self, decimals=0, out=None):
        return Point(np.round(self.p, decimals, out))

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

    def copy(self):
        return Size(self.w, self.h)

    @property
    def w(self):
        return self.p[0]

    @property
    def h(self):
        return self.p[1]

    def astype(self, dtype):
        return Size(self.p.astype(dtype))

    def round(self, decimals=0, out=None):
        return Size(np.round(self.p, decimals, out))

    def wh(self):
        return self.w, self.h

    def hw(self):
        return self.h, self.w

    def scale(self, scale):
        return Size(self.p * scale)

    def __str__(self):
        return self.to_string()

    def __truediv__(self, other):
        return Size(self.p / other)

    def __floordiv__(self, other):
        return Size(self.p // other)

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


class Coords(SerializableType):
    def __init__(self, points: np.ndarray = np.zeros((0, 2), dtype=float)):
        self.points = np.array(points, dtype=float)

    def __str__(self):
        return self.to_string()

    def scale(self, factor):
        return Coords(self.points * factor)

    def rotate(self, degree, origin):
        def rotate_point(xy, radians, origin=(0, 0)):
            x, y = xy
            offset_x, offset_y = origin
            adjusted_x, adjusted_y = (x - offset_x), (y - offset_y)
            cos_rad, sin_rad = np.cos(radians), np.sin(radians)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

            return qx, qy

        self.points = np.array([rotate_point(p, degree / 180 * np.pi, origin) for p in self.points])

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

    @classmethod
    def _deserialize(cls, value):
        return cls.from_json(value)

    def to_json(self):
        return self.to_string()

    def _serialize(self):
        return self.to_string()

    def center_y(self):
        return np.mean(self.points[:, 1])

    def interpolate_y(self, x):
        return np.interp(x, self.points[:, 0], self.points[:, 1])

    def approximate(self, distance):
        self.points = approximate_polygon(self.points, distance)

    def draw(self, canvas, color=(0, 255, 0), thickness=5, fill=False, offset=(0, 0), scale=None):
        if scale:
            points = scale(self.points)
        else:
            points = self.points
        pts = np.round((points + offset).reshape((-1, 1, 2))).astype(np.int32)
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
        tl_y = int(aabb.tl.y)
        br_y= int(aabb.br.y)
        tl_x = int(aabb.tl.x)
        br_x = int(aabb.br.x)
        sub_image = image[tl_y:br_y, tl_x:br_x]
        return sub_image

    def to_points_list(self):
        points = []
        for x in self.points:
            points.append((x[0], x[1]))
        return points

class Rect:
    def __init__(self, origin: Point = None, size: Union[Point, Size] = None):
        self.origin = origin if origin else Point()
        if size and isinstance(size, Point):
            self.size = Size(size - origin)
        else:
            self.size = size if size else Size()

    def to_coords(self) -> Coords:
        return Coords(np.array([self.origin.p, self.origin.p + (self.size.w, 0),
                                self.origin.p + self.size.p, self.origin.p + (0, self.size.h)]))

    def union(self, aabb: 'Rect') -> 'Rect':
        if aabb.area() == 0:
            return self
        elif self.area() == 0:
            return aabb

        top = min(aabb.top(), self.top())
        left = min(aabb.left(), self.left())
        bottom = max(aabb.bottom(), self.bottom())
        right = max(aabb.right(), self.right())
        return Rect(Point(np.array([left, top])), Size(np.array([right - left, bottom - top])))

    @property
    def tl(self):
        return self.origin

    @property
    def br(self):
        return self.origin + self.size

    def left(self):
        return self.origin.x

    def right(self):
        return self.origin.x + self.size.w

    def top(self):
        return self.origin.y

    def bottom(self):
        return self.origin.y + self.size.h

    def area(self):
        return self.size.h * self.size.w

    @property
    def center(self):
        return self.origin + self.size / 2

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

    def copy(self) -> 'Rect':
        return Rect(self.origin.copy(), self.size.copy())


if __name__ == '__main__':
    c = Coords(np.array([[0, 1], [1, 2], [6, -123]]))
    print(c.to_json())
    print(Coords.from_json(c.to_json()).to_json() == c.to_json())
    print(c.aabb())

    p = Point(-20, 100).astype(float)
    print(p.to_json())
    print(Point.from_json(p.to_json()).to_json() == p.to_json())
