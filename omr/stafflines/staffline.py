import numpy as np
from typing import List
import cv2
from skimage.measure import approximate_polygon


class StaffLine:
    @staticmethod
    def from_json(line):
        line = line['line']
        return StaffLine([(p['x'], p['y']) for p in line['points']])

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points)
        self.approx_line = self.points
        self._center_y = np.mean(self.points[:, 1])
        self._dewarped_y = int(self._center_y)

    def approximate(self, distance):
        self.approx_line = approximate_polygon(self.points, distance)

    def interpolate_y(self, x):
        return np.interp(x, self.approx_line[:, 0], self.approx_line[:, 1])

    def center_y(self):
        return self._center_y

    def dewarped_y(self):
        return self._dewarped_y

    def draw(self, canvas, color=(0, 255, 0), thickness=5):
        # pts = self.points.reshape((-1, 1, 2)).astype(np.int32)
        pts = self.approx_line.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, int(thickness))

    def json(self):
        out = {
            'points': []
        }
        for x, y in self.approx_line:
            out['points'].append({'x': int(x), 'y': int(y)})

        return {'line': out}


class Staff:
    @staticmethod
    def from_json(staff):
        return Staff([StaffLine.from_json(line) for line in staff['lines']])

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

    def draw(self, canvas, color=(0, 255, 0), thickness=5):
        for line in self.staff_lines:
            line.draw(canvas, color, thickness)

    def json(self):
        out = {'lines': []}
        for line in self.staff_lines:
            out['lines'].append(line.json())

        return out


class Staffs:
    @staticmethod
    def from_json(obj: dict):
        return Staffs([Staff.from_json(jstaff) for jstaff in obj['staffs']] if 'staffs' in obj else [],
                      crop=tuple(obj['crop']) if 'crop' in 'staffs' else (0, 0, -1, -1),
                      approx=False)

    def __init__(self, staffs: List[Staff], crop=(0, 0, -1, -1), approx=True):
        self.staffs = staffs
        self.crop = crop

        self.avg_staff_line_distance = np.mean([v for v in [d.avg_staff_line_distance for d in self.staffs] if v > 0])
        self.avg_staff_line_distance = max([5, self.avg_staff_line_distance])
        if approx:
            self.approximate()

    def avg_staff_distance(self):
        d = []
        for i in range(1, len(self.staffs)):
            top = self.staffs[i - 1].staff_lines[-1].center_y()
            bot = self.staffs[i].staff_lines[0].center_y()
            d.append(bot - top)

        return np.mean(d)

    def approximate(self):
        for staff in self.staffs:
            staff.approximate(self.avg_staff_line_distance / 10)

    def draw(self, canvas, color=(0, 255, 0), thickness=-1):
        if thickness < 0:
            thickness = self.avg_staff_line_distance / 10 if self.avg_staff_line_distance > 0 else 5

        for staff in self.staffs:
            staff.draw(canvas, color, thickness)

    def json(self):
        return {
            'staffs': [s.json() for s in self.staffs],
            'crop': list(map(int, self.crop)),
        }







