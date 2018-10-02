from . import Coords, Point, Syllable
from typing import List
from enum import Enum


class AccidentalType(Enum):
    NATURAL = 0
    SHARP = 1
    FLAT = -1


class MusicSymbolPositionInStaff(Enum):
    UNDEFINED = -1

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
    NORMAL = 'normal'


class GraphicalConnectionType(Enum):
    NONE = 'none'
    CONNECTED = 'connected'


class Note:
    def __init__(self, note_type=NoteType.NORMAL,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 graphical_connection=GraphicalConnectionType.NONE,
                 accidental: Accidental = None,
                 syllable: Syllable = None,
                 index = 0,
                 ):
        self.note_type = note_type
        self.coord = coord
        self.position_in_staff = position_in_staff
        self.graphical_connection = graphical_connection
        self.accidental = accidental
        self.syllable = syllable
        self.index = 0

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
            MusicSymbolPositionInStaff(json.get('position_in_staff', MusicSymbolPositionInStaff.UNDEFINED)),
            GraphicalConnectionType(json.get('graphical_connection', GraphicalConnectionType.NONE)),
            Accidental.from_json(json.get('accidental', {})),
            json.get('syllable', None),
            json.get('index', 0),
        )

    def to_json(self):
        return {
            'type': self.note_type.value,
            'coord': self.coord.to_json(),
            'position_in_staff': self.position_in_staff.value,
            'graphical_connection': self.graphical_connection.value,
            'accidental': self.accidental.to_json(),
            'syllable': self.syllable.id,
            'index': self.index,
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
            Coords.from_json(json.get("coords", [])),
            MusicSymbolPositionInStaff(json.get('position_in_staff', MusicSymbolPositionInStaff.UNDEFINED)),
        )

    def to_json(self):
        return {
            "type": self.clef_type.value,
            "coord": self.coord.to_json(),
            "position_in_staff": self.position_in_staff.value,
        }


class StaffLine:
    def __init__(self, coords=Coords()):
        self.coords = coords

    @staticmethod
    def from_json(json):
        return StaffLine(
            Coords.from_json(json.get('coords', []))
        )

    def to_json(self):
        return {
            'coords': self.coords.to_json()
        }


class StaffEquiv:
    def __init__(self,
                 coords=Coords(),
                 staff_lines: List[StaffLine]=list(),
                 clefs: List[Clef]=list(),
                 notes: List[Note]=list()):
        self.coords = coords
        self.staff_lines = staff_lines
        self.clefs = clefs
        self.notes = notes

    def _resolve_cross_refs(self, page):
        for note in self.notes:
            note._resolve_cross_refs(page)

    @staticmethod
    def from_json(json):
        return StaffEquiv(
            Coords.from_json(json.get('coords', [])),
            [StaffLine.from_json(l) for l in json.get('staff_lines', [])],
            [Clef.from_json(c) for c in json.get('clefs', [])],
            [Note.from_json(n) for n in json.get('note', [])]
        )

    def to_json(self):
        return {
            "coords": self.coords.to_json(),
            "staff_lines": [l.to_json() for l in self.staff_lines],
            'clefs': [c.to_json() for c in self.clefs],
            "notes": [n.to_json() for n in self.notes],
        }
