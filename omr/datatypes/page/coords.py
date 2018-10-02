import numpy as np


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


if __name__ == '__main__':
    c = Coords(np.array([[0, 1], [1, 2], [6, -123]]))
    print(c.to_json())
    print(Coords.from_json(c.to_json()).to_json() == c.to_json())

    p = Point(-20, 100)
    print(p.to_json())
    print(Point.from_json(p.to_json()).to_json() == p.to_json())
