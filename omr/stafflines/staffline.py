import numpy as np
from typing import List
import cv2

class StaffLine:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.approx_line = self.points

    def approximate(self, distance):
        pass

    def center_y(self):
        return np.mean(self.points[:, 1])

    def draw(self, canvas):
        pts = self.points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 0), 5)


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


class Staffs:
    def __init__(self, staffs: List[Staff]):
        self.staffs = staffs

        self.avg_staff_line_distance = np.mean([v for v in [d.avg_staff_line_distance for d in self.staffs] if v > 0])
        self.avg_staff_line_distance = max([5, self.avg_staff_line_distance])
        self.approximate()

    def approximate(self):
        for staff in self.staffs:
            staff.approximate(self.avg_staff_line_distance)

    def draw(self, canvas):
        for staff in self.staffs:
            staff.draw(canvas)







