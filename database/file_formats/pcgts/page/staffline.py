from .coords import Coords, Rect, Point
from typing import List, Tuple
import numpy as np
from .definitions import MusicSymbolPositionInStaff
from dataclasses import dataclass


class StaffLine:
    def __init__(self, coords=Coords(), highlighted=False, space=False, dry_point_line=False, sl_id=''):
        self.id = sl_id
        self.coords = coords
        self._center_y = 0
        self._dewarped_y = 0
        self.highlighted = highlighted
        self.dry_point_line = dry_point_line
        self.space = space
        self.update()

    @staticmethod
    def from_json(json):
        return StaffLine(
            Coords.from_json(json.get('coords', [])),
            json.get('highlighted', False),
            json.get('space', False),
            json.get('dry_point_line'),
            json.get('id', ''),
        )

    def to_json(self):
        return {
            'id': self.id,
            'coords': self.coords.to_json(),
            'highlighted': self.highlighted,
            'space': self.space,
            'dry_point_line': self.dry_point_line,
        }

    def update(self):
        self._center_y = np.mean(self.coords.points[:, 1])
        self._dewarped_y = self._center_y

    def approximate(self, distance):
        self.coords.approximate(distance)
        self.update()

    def interpolate_y(self, x):
        return self.coords.interpolate_y(x)

    def center_y(self):
        return self._center_y

    def dewarped_y(self):
        return self._dewarped_y

    def draw(self, canvas, color=(0, 255, 0), thickness=5, offset=(0, 0), scale=None):
        self.coords.draw(canvas, color, thickness, offset=offset, scale=scale)

    def fit_to_gray_image(self, gray: np.ndarray, offset=5, debug=False):
        # bounds
        left, top = tuple(list(map(int, self.coords.points.min(axis=0))))
        right, bot = tuple(list(map(int, self.coords.points.max(axis=0))))

        # padding
        gray = np.pad(gray, ((2 * offset, 2 * offset), (2 * offset, 2 * offset)), mode='constant', constant_values=0)
        left += 2 * offset
        top += 2 * offset
        right += 2 * offset
        bot += 2 * offset

        # offset
        top -= offset
        bot += offset

        # lines
        line = np.zeros(gray.shape)
        self.draw(line, color=(255,), thickness=2, offset=(2 * offset, 2 * offset))
        target = gray[top - offset:bot + offset, left:right]
        search = line[top:bot, left:right]

        fit = [np.mean(target[i:i + bot - top, :] * search) for i in range(offset * 2)]
        shift = np.argmin(fit) - offset
        self.coords.points[:, 1] += shift

        # debug output
        if debug:
            import matplotlib.pyplot as plt
            sub_imgs = [target[i:i + bot - top, :] * search for i in range(offset * 2)]
            f, ax = plt.subplots(len(sub_imgs), 1)
            for a, si in zip(ax, sub_imgs):
                a.imshow(si)
            plt.show()
            print(shift)


class StaffLines(List[StaffLine]):
    @staticmethod
    def from_json(json):
        return StaffLines([StaffLine.from_json(l) for l in json]).sorted()

    def to_json(self):
        return [l.to_json() for l in self]

    def draw(self, canvas, color=(0, 255, 0), thickness=5, scale=None):
        for l in self:
            l.draw(canvas, color, thickness, scale=scale)

    def aabb(self) -> Rect:
        if len(self) == 0:
            return Rect()
        r = self[0].coords.aabb()
        for sl in self[1:]:
            r = r.union(sl.coords.aabb())
        return r

    def sort(self):
        super(StaffLines, self).sort(key=lambda s: s.center_y())

    def sorted(self):
        return StaffLines(sorted(self, key=lambda s: s.center_y()))

    def compute_position_in_staff(self, coord: Point, clef=False) -> MusicSymbolPositionInStaff:
        return self.position_in_staff(coord, clef)

    def compute_coord_by_position_in_staff(self, x: float, pis: MusicSymbolPositionInStaff) -> Point:
        line = pis.value - MusicSymbolPositionInStaff.LINE_1
        if line < 0:
            return Point(x, self[-1].interpolate_y(x) + abs(line) / 2 * self.avg_line_distance())
        elif line // 2 + 1 >= len(self):
            return Point(x, self[0].interpolate_y(x) - abs(len(self) - 1 - line / 2) * self.avg_line_distance())
        elif line % 2 == 0:
            return Point(x, self[len(self) - 1 - line // 2].interpolate_y(x))
        else:
            return Point(x, (self[len(self) - 1 - line // 2].interpolate_y(x) + self[
                len(self) - line // 2 - 2].interpolate_y(x)) / 2)

    def avg_line_distance(self, default=-1):
        if len(self) <= 1:
            return default

        ys = [sl.center_y() for sl in self]
        d = max(ys) - min(ys)
        return d / (len(self) - 1)

    # Following code taken from ommr4all-client
    # ==================================================================
    @staticmethod
    def _round_to_staff_pos(x: float):
        rounded = np.round(x)
        even = (rounded + 2000) % 2 == 0
        if not even:
            if abs(x - rounded) < 0.4:
                return rounded
            else:
                return rounded + 1 if x - rounded > 0 else rounded - 1
        else:
            return rounded

    @staticmethod
    def _interp_staff_pos(y: float, top: float, bot: float, top_space: bool, bot_space: bool,
                          top_pos: MusicSymbolPositionInStaff, bot_pos: MusicSymbolPositionInStaff,
                          offset: int, clef=False) -> Tuple[float, MusicSymbolPositionInStaff]:
        ld = bot - top
        if top_space and not bot_space:
            top -= ld
            top_pos += 1
        elif not top_space and bot_space:
            bot += ld
            bot_pos -= 1
        elif top_space and bot_space:
            center = (top + bot) / 1
            if center > y:
                top -= ld / 2
                bot = center
                top_pos += 1
                bot_pos = top_pos - 2
            else:
                top = center
                bot += ld / 2
                top_pos -= 1
                top_pos = bot_pos + 2

        d = y - top
        rel = d / (bot - top)
        snapped = -offset + StaffLines._round_to_staff_pos(2 * rel)
        pis = int(top_pos - snapped)
        if clef:
            if pis % 2 != 1:
                pis = pis + 1

        return top + snapped * (bot - top) / 2, \
               MusicSymbolPositionInStaff(max(MusicSymbolPositionInStaff.SPACE_0,
                                              min(MusicSymbolPositionInStaff.SPACE_7, pis)))

    def _staff_pos(self, p: Point, offset: int = 0, clef=False) -> Tuple[float, MusicSymbolPositionInStaff]:
        @dataclass
        class StaffPos:
            line: StaffLine
            y: float
            pos: MusicSymbolPositionInStaff

        if len(self) <= 1:
            return p.y, MusicSymbolPositionInStaff.UNDEFINED

        y_on_staff: List[StaffPos] = []
        for staffLine in self.sorted():
            y_on_staff.append(
                StaffPos(staffLine, staffLine.coords.interpolate_y(p.x), MusicSymbolPositionInStaff.UNDEFINED))

        y_on_staff[-1].pos = MusicSymbolPositionInStaff.SPACE_1 if y_on_staff[
            -1].line.space else MusicSymbolPositionInStaff.LINE_1
        for i in reversed(range(0, len(y_on_staff) - 1)):
            if y_on_staff[i + 1].line.space == y_on_staff[i].line.space:
                y_on_staff[i].pos = y_on_staff[i + 1].pos + 2
            else:
                y_on_staff[i].pos = y_on_staff[i + 1].pos + 1

        pre_line_idx = -1
        l = [i for i, l in enumerate(y_on_staff) if l.y > p.y]
        if len(l) > 0:
            pre_line_idx = l[0]

        if pre_line_idx == -1:
            # bot
            last = y_on_staff[-1]
            prev = y_on_staff[-2]
        elif pre_line_idx == 0:
            last = y_on_staff[pre_line_idx + 1]
            prev = y_on_staff[pre_line_idx]
        else:
            last = y_on_staff[pre_line_idx]
            prev = y_on_staff[pre_line_idx - 1]

        return StaffLines._interp_staff_pos(p.y, prev.y, last.y, prev.line.space, last.line.space, prev.pos, last.pos,
                                            offset, clef)

    def position_in_staff(self, p: Point, clef=False) -> MusicSymbolPositionInStaff:
        return self._staff_pos(p, clef=clef)[1]

    def snap_to_pos(self, p: Point, offset: int = 0) -> float:
        return self._staff_pos(p, offset)[0]

    # ==================================================================
