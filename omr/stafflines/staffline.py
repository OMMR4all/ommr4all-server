import numpy as np
from typing import List
import cv2
from skimage.measure import approximate_polygon

class StaffLine:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points)
        self.approx_line = self.points

    def approximate(self, distance):
        self.approx_line = approximate_polygon(self.points, distance)

    def center_y(self):
        return np.mean(self.points[:, 1])

    def draw(self, canvas):
        # pts = self.points.reshape((-1, 1, 2)).astype(np.int32)
        pts = self.approx_line.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 0), 5)

    def json(self):
        out = {
            'points': []
        }
        for x, y in self.approx_line:
            out['points'].append({'x': x, 'y': y})

        return {'line': out}


class Staff:
    def __init__(self, staff_lines: List[StaffLine]):
        self.staff_lines = staff_lines
        self.avg_staff_line_distance = self._avg_line_distance()

    def _avg_line_distance(self, default=-1):
        if len(self.staff_lines) <= 1:
            return default

        d = self.staff_lines[-1].center_y() - self.staff_lines[0].center_y()
        return d / (len(self.staff_lines) - 1)

    def approximate(self, distance):
        for line in self.staff_lines:
            line.approximate(distance)

    def draw(self, canvas):
        for line in self.staff_lines:
            line.draw(canvas)

    def json(self):
        out = {'lines': []}
        for line in self.staff_lines:
            out['lines'].append(line.json())

        return out


class Staffs:
    def __init__(self, staffs: List[Staff]):
        self.staffs = staffs

        self.avg_staff_line_distance = np.mean([v for v in [d.avg_staff_line_distance for d in self.staffs] if v > 0])
        self.avg_staff_line_distance = max([5, self.avg_staff_line_distance])
        self.approximate()

    def approximate(self):
        for staff in self.staffs:
            staff.approximate(self.avg_staff_line_distance / 10)

    def draw(self, canvas):
        for staff in self.staffs:
            staff.draw(canvas)

    def json(self):
        out = {'staffs': []}
        for staff in self.staffs:
            out['staffs'].append(staff.json())

        return out







