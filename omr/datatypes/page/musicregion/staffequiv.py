from . import Coords, Point, Syllable, EquivIndex
from typing import List
from enum import Enum
from skimage.measure import approximate_polygon
import cv2
import numpy as np


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


class Accidental:
    def __init__(self,
                 accidental=AccidentalType.NATURAL,
                 coord=Point()):
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
        return {
            'type': self.accidental.value,
            'coord': self.coord.to_json()
        }


class NoteType(Enum):
    NORMAL = 0


class GraphicalConnectionType(Enum):
    NONE = 0
    CONNECTED = 1


class Note:
    def __init__(self, note_type=NoteType.NORMAL,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 graphical_connection=GraphicalConnectionType.NONE,
                 accidental: Accidental = None,
                 syllable: Syllable = None,
                 ):
        self.note_type = note_type
        self.coord = coord
        self.position_in_staff = position_in_staff
        self.graphical_connection = graphical_connection
        self.accidental = accidental
        self.syllable = syllable

    def _resolve_cross_refs(self, page):
        if self.syllable is not None and not isinstance(self.syllable, Syllable):
            # resolve cross ref
            s_id = self.syllable
            self.syllable = page.syllable_by_id(s_id)
            if self.syllable is None:
                raise Exception("Syllable id '{}' not found!".format(s_id))

    @staticmethod
    def from_json(json: dict):
        return Note(
            NoteType(json.get('type', NoteType.NORMAL)),
            Coords.from_json(json.get('coord', [])),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
            GraphicalConnectionType(json.get('graphicalConnection', GraphicalConnectionType.NONE)),
            Accidental.from_json(json.get('accidental', {})),
            json.get('syllable', None),
        )

    def to_json(self):
        return {
            'type': self.note_type.value,
            'coord': self.coord.to_json(),
            'positionInStaff': self.position_in_staff.value,
            'graphicalConnection': self.graphical_connection.value,
            'accidental': self.accidental.to_json() if self.accidental else None,
            'syllable': self.syllable.id if self.syllable else None,
        }


class ClefType(Enum):
    CLEF_F = 0
    CLEF_C = 1


class Clef:
    def __init__(self,
                 clef_type=ClefType.CLEF_F,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED):
        self.clef_type = clef_type
        self.coord = coord
        self.position_in_staff = position_in_staff

    @staticmethod
    def from_json(json: dict):
        return Clef(
            ClefType(json.get("type", ClefType.CLEF_F)),
            Coords.from_json(json.get("coord", "0,0")),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
        )

    def to_json(self):
        return {
            "type": self.clef_type.value,
            "coord": self.coord.to_json(),
            "positionInStaff": self.position_in_staff.value,
        }


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

    def draw(self, canvas, color=(0, 255, 0), thickness=5):
        self.coords.draw(canvas, color, thickness)


class StaffEquiv:
    def __init__(self,
                 coords=Coords(),
                 staff_lines: List[StaffLine]=None,
                 clefs: List[Clef]=None,
                 notes: List[Note]=None,
                 index=EquivIndex.CORRECTED):
        self.coords = coords
        self.staff_lines = staff_lines if staff_lines else []
        self.clefs = clefs if clefs else []
        self.notes = notes if notes else []
        self.index = index

    def _resolve_cross_refs(self, page):
        for note in self.notes:
            note._resolve_cross_refs(page)

    @staticmethod
    def from_json(json):
        return StaffEquiv(
            Coords.from_json(json.get('coords', [])),
            [StaffLine.from_json(l) for l in json.get('staffLines', [])],
            [Clef.from_json(c) for c in json.get('clefs', [])],
            [Note.from_json(n) for n in json.get('notes', [])],
            EquivIndex(json.get('index', EquivIndex.CORRECTED)),
        )

    def to_json(self):
        return {
            "coords": self.coords.to_json(),
            "staffLines": [l.to_json() for l in self.staff_lines],
            'clefs': [c.to_json() for c in self.clefs],
            "notes": [n.to_json() for n in self.notes],
            'index': self.index.value,
        }

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
