from . import Coords, Point, Syllable, EquivIndex
from typing import List, Tuple
from enum import Enum
from skimage.measure import approximate_polygon
import cv2
import numpy as np
from abc import ABC, abstractmethod
from uuid import uuid4


class SymbolType(Enum):
    NEUME = 0
    CLEF = 1
    ACCID = 2

    NOTE_COMPONENT = 3


class Symbol(ABC):
    def __init__(self, symbol_type: SymbolType):
        self.symbol_type = symbol_type

    @abstractmethod
    def to_json(self):
        return {
            'symbol': self.symbol_type.value,
        }

    @staticmethod
    def from_json(json: dict):
        return {
            SymbolType.NEUME: Neume.from_json,
            SymbolType.CLEF: Clef.from_json,
            SymbolType.ACCID: Accidental.from_json,
        }[SymbolType(json.get('symbol'))](json)


class AccidentalType(Enum):
    NATURAL = 0
    SHARP = 1
    FLAT = -1


class MusicSymbolPositionInStaff(Enum):
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
                 coord=Point()):
        super().__init__(SymbolType.ACCID)
        self.accidental = accidental
        self.coord = coord

    @staticmethod
    def from_json(json):
        if json is None:
            return None
        return Accidental(
            AccidentalType(json.get('type', AccidentalType.NATURAL)),
            Point.from_json(json.get('coord', Point().to_json())),
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'type': self.accidental.value,
            'coord': self.coord.to_json()
        })


class NoteType(Enum):
    NORMAL = 0


class GraphicalConnectionType(Enum):
    GAPED = 0
    LOOPED = 1


class NoteName(Enum):
    UNDEFINED = -1
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6


class NoteComponent(Symbol):
    def __init__(self,
                 note_name=NoteName.UNDEFINED,
                 octave=-1,
                 note_type=NoteType.NORMAL,
                 coord: Point = None,
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 graphical_connection=GraphicalConnectionType.GAPED,
                 ):
        super().__init__(SymbolType.NOTE_COMPONENT)
        self.note_name = note_name
        self.octave = octave
        self.note_type = note_type
        self.coord: Point = coord if coord else Point()
        self.position_in_staff = position_in_staff
        self.graphical_connection = graphical_connection

    @staticmethod
    def from_json(json: dict):
        return NoteComponent(
            NoteName(json.get('pname', NoteName.UNDEFINED)),
            json.get('oct', -1),
            NoteType(json.get('type', NoteType.NORMAL)),
            Point.from_json(json.get('coord', [])),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
            GraphicalConnectionType(json.get('graphicalConnection', GraphicalConnectionType.GAPED)),
        )

    def to_json(self):
        return {
            'pname': self.note_name.value,
            'oct': self.octave,
            'type': self.note_type.value,
            'coord': self.coord.to_json(),
            'positionInStaff': self.position_in_staff.value,
            'graphicalConnection': self.graphical_connection.value,
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
            [NoteComponent.from_json(nc) for nc in json.get('nc', [])]
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'id': self.id,
            'nc': [nc.to_json() for nc in self.notes]
        })


class ClefType(Enum):
    CLEF_F = 0
    CLEF_C = 1


class Clef(Symbol):
    def __init__(self,
                 clef_type=ClefType.CLEF_F,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED):
        super().__init__(SymbolType.CLEF)
        self.clef_type = clef_type
        self.coord = coord
        self.position_in_staff = position_in_staff

    @staticmethod
    def from_json(json: dict):
        return Clef(
            ClefType(json.get("type", ClefType.CLEF_F)),
            Point.from_json(json.get("coord", "0,0")),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
        )

    def to_json(self):
        return dict(super().to_json(), **{
            "type": self.clef_type.value,
            "coord": self.coord.to_json(),
            "positionInStaff": self.position_in_staff.value,
        })


class StaffLine:
    def __init__(self, coords=Coords()):
        self.coords = coords
        self._center_y = 0
        self._dewarped_y = 0
        self.update()

    @staticmethod
    def from_json(json):
        return StaffLine(
            Coords.from_json(json.get('coords', []))
        )

    def to_json(self):
        return {
            'coords': self.coords.to_json()
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

        d = self.staff_lines[-1].center_y() - self.staff_lines[0].center_y()
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


class MusicLines(List[MusicLine]):
    @staticmethod
    def from_json(json):
        return MusicLines([MusicLine.from_json(l) for l in json])

    def to_json(self):
        return [l.to_json() for l in self]

    def draw(self, canvas):
        for ml in self:
            ml.draw(canvas)

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


