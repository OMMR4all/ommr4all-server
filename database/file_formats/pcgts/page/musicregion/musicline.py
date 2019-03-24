from database.file_formats.pcgts.page.musicregion import Coords, Point
from typing import List, Tuple
from enum import IntEnum
import numpy as np
from abc import ABC, abstractmethod
from uuid import uuid4
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SymbolType(IntEnum):
    NEUME = 0
    CLEF = 1
    ACCID = 2

    NOTE_COMPONENT = 3


class Symbol(ABC):
    def __init__(self,
                 symbol_type: SymbolType,
                 fixed_sorting: bool = False,
                 ):
        self.symbol_type = symbol_type
        self.fixed_sorting = fixed_sorting

    @abstractmethod
    def to_json(self):
        return {
            'symbol': self.symbol_type.value,
            'fixedSorting': self.fixed_sorting,
        }

    @staticmethod
    def from_json(json: dict):
        return {
            SymbolType.NEUME: Neume.from_json,
            SymbolType.CLEF: Clef.from_json,
            SymbolType.ACCID: Accidental.from_json,
        }[SymbolType(json.get('symbol'))](json)


class AccidentalType(IntEnum):
    NATURAL = 0
    SHARP = 1
    FLAT = -1


class MusicSymbolPositionInStaff(IntEnum):
    UNDEFINED = -1000

    # usual notation
    SPACE_0 = 0
    LINE_0 = 1
    SPACE_1 = 2
    LINE_1 = 3
    SPACE_2 = 4
    LINE_2 = 5
    SPACE_3 = 6
    LINE_3 = 7
    SPACE_4 = 8
    LINE_4 = 9
    SPACE_5 = 10
    LINE_5 = 11
    SPACE_6 = 12
    LINE_6 = 13
    SPACE_7 = 14

    # 11th Century Notation only store up/down/equal
    UP = 101
    DOWN = 99
    EQUAL = 100

    def is_undefined(self):
        return self.value == MusicSymbolPositionInStaff.UNDEFINED

    def is_absolute(self):
        return MusicSymbolPositionInStaff.SPACE_0 <= self.value < MusicSymbolPositionInStaff.SPACE_7

    def is_relative(self):
        return MusicSymbolPositionInStaff.UP <= self.value <= MusicSymbolPositionInStaff.DOWN


class Accidental(Symbol):
    def __init__(self,
                 accidental=AccidentalType.NATURAL,
                 coord=Point(),
                 fixed_sorting: bool = False,
                 ):
        super().__init__(SymbolType.ACCID, fixed_sorting)
        self.accidental = accidental
        self.coord = coord

    @staticmethod
    def from_json(json):
        if json is None:
            return None
        return Accidental(
            AccidentalType(json.get('type', AccidentalType.NATURAL)),
            Point.from_json(json.get('coord', Point().to_json())),
            json.get('fixedSorting', False),
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'type': self.accidental.value,
            'coord': self.coord.to_json()
        })


class NoteType(IntEnum):
    NORMAL = 0


class GraphicalConnectionType(IntEnum):
    GAPED = 0
    LOOPED = 1


class NoteName(IntEnum):
    UNDEFINED = -1
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6

    def __str__(self):
        return 'ABCDEFG '[self.value]


class NoteComponent(Symbol):
    def __init__(self,
                 note_name=NoteName.UNDEFINED,
                 octave=-1,
                 note_type=NoteType.NORMAL,
                 coord: Point = None,
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 graphical_connection=GraphicalConnectionType.GAPED,
                 fixed_sorting: bool = False,
                 ):
        super().__init__(SymbolType.NOTE_COMPONENT, fixed_sorting)
        self.note_name = note_name
        self.octave = octave
        self.note_type = note_type
        self.coord: Point = coord if coord else Point()
        self.position_in_staff = position_in_staff
        self.graphical_connection = graphical_connection

    def update_note_name(self, clef):
        self.note_name, self.octave = clef.note_name_octave(self.position_in_staff)

    @staticmethod
    def from_json(json: dict):
        try:
            return NoteComponent(
                NoteName(json.get('pname', NoteName.UNDEFINED)),
                json.get('oct', -1),
                NoteType(json.get('type', NoteType.NORMAL)),
                Point.from_json(json.get('coord', [])),
                MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
                GraphicalConnectionType(json.get('graphicalConnection', GraphicalConnectionType.GAPED)),
                json.get('fixedSorting', False),
            )
        except Exception as e:
            logger.exception(e)
            logger.error("Got faulty dict {}".format(json))
            return None

    def to_json(self):
        return {
            'pname': self.note_name.value,
            'oct': self.octave,
            'type': self.note_type.value,
            'coord': self.coord.to_json(),
            'positionInStaff': self.position_in_staff.value,
            'graphicalConnection': self.graphical_connection.value,
            'fixedSorting': self.fixed_sorting,
        }


class Neume(Symbol):
    def __init__(self,
                 n_id = str(uuid4()),
                 notes: List[NoteComponent] = None,
                 ):
        super().__init__(SymbolType.NEUME)
        self.id = n_id
        self.notes: List[NoteComponent] = notes if notes else []

    @staticmethod
    def from_json(json: dict):
        return Neume(
            json.get('id', str(uuid4())),
            [nc for nc in [NoteComponent.from_json(nc) for nc in json.get('nc', [])] if nc],
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'id': self.id,
            'nc': [nc.to_json() for nc in self.notes]
        })


class ClefType(IntEnum):
    CLEF_F = 0
    CLEF_C = 1

    def offset(self):
        return [0, -3][self.value]


class Clef(Symbol):
    def __init__(self,
                 clef_type=ClefType.CLEF_F,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 fixed_sorting: bool = False,
                 ):
        super().__init__(SymbolType.CLEF, fixed_sorting)
        self.clef_type = clef_type
        self.coord = coord
        self.position_in_staff = position_in_staff

    @staticmethod
    def from_json(json: dict):
        return Clef(
            ClefType(json.get("type", ClefType.CLEF_F)),
            Point.from_json(json.get("coord", "0,0")),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
            json.get('fixedSorting', False),
        )

    def to_json(self):
        return dict(super().to_json(), **{
            "type": self.clef_type.value,
            "coord": self.coord.to_json(),
            "positionInStaff": self.position_in_staff.value,
        })

    def note_name_octave(self, position_in_staff: MusicSymbolPositionInStaff):
        clef_type_offset = self.clef_type.offset()
        note_name = NoteName((clef_type_offset + 49 - self.position_in_staff + MusicSymbolPositionInStaff.LINE_2 + position_in_staff) % 7)
        octave = 4 + (clef_type_offset - self.position_in_staff + MusicSymbolPositionInStaff.LINE_1 + position_in_staff) // 7
        return note_name, octave


class StaffLine:
    def __init__(self, coords=Coords(), highlighted=False, space=False, sl_id=''):
        self.id = sl_id
        self.coords = coords
        self._center_y = 0
        self._dewarped_y = 0
        self.highlighted = highlighted
        self.space = space
        self.update()

    @staticmethod
    def from_json(json):
        return StaffLine(
            Coords.from_json(json.get('coords', [])),
            json.get('highlighted', False),
            json.get('space', False),
            json.get('id', ''),
        )

    def to_json(self):
        return {
            'id': self.id,
            'coords': self.coords.to_json(),
            'highlighted': self.highlighted,
            'space': self.space,
        }

    def update(self):
        self._center_y = np.mean(self.coords.points[:, 1])
        self._dewarped_y = int(self._center_y)

    def approximate(self, distance):
        self.coords.approximate(distance)
        self.update()

    def interpolate_y(self, x):
        return self.coords.interpolate_y(x)

    def center_y(self):
        return self._center_y

    def dewarped_y(self):
        return self._dewarped_y

    def draw(self, canvas, color=(0, 255, 0), thickness=5, offset=(0, 0)):
        self.coords.draw(canvas, color, thickness, offset=offset)

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
        target = gray[top-offset:bot+offset, left:right]
        search = line[top:bot, left:right]

        fit = [np.mean(target[i:i+bot-top, :] * search) for i in range(offset * 2)]
        shift = np.argmin(fit) - offset
        self.coords.points[:, 1] += shift

        # debug output
        if debug:
            import matplotlib.pyplot as plt
            sub_imgs = [target[i:i+bot-top, :] * search for i in range(offset * 2)]
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

    def draw(self, canvas, color=(0, 255, 0), thickness=5):
        for l in self:
            l.draw(canvas, color, thickness)

    def sort(self):
        super(StaffLines, self).sort(key=lambda s: s.center_y())

    def sorted(self):
        return StaffLines(sorted(self, key=lambda s: s.center_y()))

    def compute_position_in_staff(self, coord: Point) -> MusicSymbolPositionInStaff:
        return self.position_in_staff(coord)

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
                          offset: int) -> Tuple[float, MusicSymbolPositionInStaff]:
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

        d = y -top
        rel = d / (bot - top)
        snapped = -offset + StaffLines._round_to_staff_pos(2 * rel)
        return top + snapped * (bot - top) / 2, MusicSymbolPositionInStaff(int(top_pos - snapped))

    def _staff_pos(self, p: Point, offset: int = 0) -> Tuple[float, MusicSymbolPositionInStaff]:
        @dataclass
        class StaffPos:
            line: StaffLine
            y: float
            pos: MusicSymbolPositionInStaff

        if len(self) <= 1:
            return p.y, MusicSymbolPositionInStaff.UNDEFINED

        y_on_staff: List[StaffPos] = []
        for staffLine in self.sorted():
            y_on_staff.append(StaffPos(staffLine, staffLine.coords.interpolate_y(p.x), MusicSymbolPositionInStaff.UNDEFINED))

        y_on_staff[-1].pos = MusicSymbolPositionInStaff.SPACE_1 if y_on_staff[-1].line.space else MusicSymbolPositionInStaff.LINE_1
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

        return StaffLines._interp_staff_pos(p.y, prev.y, last.y, prev.line.space, last.line.space, prev.pos, last.pos, offset)

    def position_in_staff(self, p: Point) -> MusicSymbolPositionInStaff:
        return self._staff_pos(p)[1]

    def snap_to_pos(self, p: Point, offset: int = 0) -> float:
        return self._staff_pos(p, offset)[0]

    # ==================================================================



class MusicLine:
    def __init__(self,
                 ml_id: str = None,
                 coords: Coords = None,
                 staff_lines: StaffLines = None,
                 symbols: List[Symbol] = None,
                 ):
        self.id = ml_id if ml_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.staff_lines = staff_lines if staff_lines else StaffLines()
        self.symbols = symbols if symbols else []
        assert(isinstance(self.coords, Coords))
        assert(isinstance(self.id, str))
        assert(isinstance(self.staff_lines, StaffLines))
        assert(isinstance(self.symbols, list))

    @staticmethod
    def from_json(json):
        return MusicLine(
            json.get('id', str(uuid4())),
            Coords.from_json(json.get('coords', [])),
            StaffLines.from_json(json.get('staffLines', [])),
            [Symbol.from_json(s) for s in json.get('symbols', [])],
        )

    def to_json(self):
        return {
            "id": self.id,
            "coords": self.coords.to_json(),
            "staffLines": self.staff_lines.to_json(),
            "symbols": [s.to_json() for s in self.symbols],
        }

    def avg_line_distance(self, default=-1):
        if len(self.staff_lines) <= 1:
            return default

        ys = [sl.center_y() for sl in self.staff_lines]
        d = max(ys) - min(ys)
        return d / (len(self.staff_lines) - 1)

    def approximate(self, distance):
        for line in self.staff_lines:
            line.approximate(distance)

    def fit_to_gray_image(self, gray: np.ndarray, offset=5):
        for line in self.staff_lines:
            line.fit_to_gray_image(gray, offset)

    def draw(self, canvas, color=(0, 255, 0), thickness=1):
        self.staff_lines.draw(canvas, color, thickness)

    def extract_image_and_gt(self, page: np.ndarray) -> (np.ndarray, str):
        image = None
        if len(self.coords.points) > 2:
            image = self.coords.extract_from_image(page)

        return image, None

    def center_y(self):
        if len(self.staff_lines) == 0:
            return None

        return (self.staff_lines[0].center_y() + self.staff_lines[-1].center_y()) / 2

    def compute_position_in_staff(self, coord: Point) -> MusicSymbolPositionInStaff:
        return self.staff_lines.compute_position_in_staff(coord)

    def update_note_names(self, initial_clef: Clef = None):
        current_clef = initial_clef if initial_clef else Clef()

        for s in self.symbols:
            if isinstance(s, Clef):
                current_clef = s
            elif isinstance(s, Accidental):
                pass
            elif isinstance(s, Neume):
                for n in s.notes:
                    n.update_note_name(current_clef)
            else:
                raise TypeError(s)

        return current_clef


class MusicLines(List[MusicLine]):
    @staticmethod
    def from_json(json):
        mls = MusicLines([MusicLine.from_json(l) for l in json])
        current_clef = None
        for ml in mls:
            current_clef = ml.update_note_names(current_clef)
        return mls

    def to_json(self):
        return [l.to_json() for l in self]

    def draw(self, canvas, color=(0, 255, 0)):
        for ml in self:
            ml.draw(canvas, color=color)

    def extract_images_and_gt(self, page: np.ndarray) -> List[Tuple[MusicLine, np.ndarray, str]]:
        for ml in self:
            img, gt = ml.extract_image_and_gt(page)
            yield ml, img, gt

    def approximate_staff_lines(self):
        d = np.mean([ml.avg_line_distance(default=0) for ml in self]) / 10
        for ml in self:
            ml.approximate(d)

    def fit_staff_lines_to_gray_image(self, gray: np.ndarray, offset=5):
        d = max(5, int(np.mean([ml.avg_line_distance(default=0) for ml in self]) / 5))
        for line in self:
            line.fit_to_gray_image(gray, d)

    def all_staff_lines(self) -> List[StaffLine]:
        return [staff_line for ml in self for staff_line in ml.staff_lines]


