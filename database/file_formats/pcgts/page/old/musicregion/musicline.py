from database.file_formats.pcgts import SymbolType, Point, MusicSymbolPositionInStaff, NoteName, \
    GraphicalConnectionType, StaffLines, NoteType, StaffLine
from database.file_formats.pcgts.page import BasicNeumeType
from database.file_formats.pcgts.page.coords import Rect, Coords
from typing import List, Tuple, Optional
from enum import IntEnum, Enum
import numpy as np
from abc import ABC, abstractmethod
from uuid import uuid4
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)



class Symbol(ABC):
    def __init__(self,
                 s_id: Optional[str],
                 symbol_type: SymbolType,
                 fixed_sorting: bool = False,
                 coord: Point = None,
                 position_in_staff: MusicSymbolPositionInStaff = MusicSymbolPositionInStaff.UNDEFINED,
                 ):
        self.id = s_id if s_id else str(uuid4())
        self.coord: Point = coord if coord else Point()
        self.symbol_type = symbol_type
        self.fixed_sorting = fixed_sorting
        self.position_in_staff = position_in_staff
        self.octave = 0
        self.note_name = NoteName.UNDEFINED

    def update_note_name(self, clef, staff_lines: Optional['StaffLines'] = None):
        if self.position_in_staff == MusicSymbolPositionInStaff.UNDEFINED:
            if staff_lines:
                self.position_in_staff = staff_lines.position_in_staff(self.coord)

        self.note_name, self.octave = clef.note_name_octave(self.position_in_staff)

    @abstractmethod
    def to_json(self):
        return {
            'id': self.id,
            'symbol': self.symbol_type.value,
            'fixedSorting': self.fixed_sorting,
            'coord': self.coord.to_json(),
            'positionInStaff': self.position_in_staff,
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


class Accidental(Symbol):
    def __init__(self,
                 accidental=AccidentalType.NATURAL,
                 coord=Point(),
                 fixed_sorting: bool = False,
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 s_id: str = None
                 ):
        super().__init__(s_id, SymbolType.ACCID, fixed_sorting, coord, position_in_staff)
        self.accidental = accidental

    @staticmethod
    def from_json(json):
        if json is None:
            return None
        return Accidental(
            AccidentalType(json.get('type', AccidentalType.NATURAL)),
            Point.from_json(json.get('coord', Point().to_json())),
            json.get('fixedSorting', False),
            json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED),
            json.get('id', None),
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'type': self.accidental.value,
        })


class NoteComponent(Symbol):
    def __init__(self,
                 note_name=NoteName.UNDEFINED,
                 octave=-1,
                 note_type=NoteType.NORMAL,
                 coord: Point = None,
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 graphical_connection=GraphicalConnectionType.GAPED,
                 fixed_sorting: bool = False,
                 s_id: str = None
                 ):
        super().__init__(s_id, SymbolType.NOTE_COMPONENT, fixed_sorting, coord, position_in_staff)
        self.note_name = note_name
        self.octave = octave
        self.note_type = note_type
        self.graphical_connection = graphical_connection

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
                json.get('id', None),
            )
        except Exception as e:
            logger.exception(e)
            logger.error("Got faulty dict {}".format(json))
            return None

    def to_json(self):
        return {
            'id': self.id,
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
                 n_id=None,
                 notes: List[NoteComponent] = None,
                 ):
        super().__init__(n_id, SymbolType.NEUME)
        self.notes: List[NoteComponent] = notes if notes else []

    @staticmethod
    def from_json(json: dict):
        return Neume(
            json.get('id', str(uuid4())),
            [nc for nc in [NoteComponent.from_json(nc) for nc in json.get('nc', [])] if nc],
        )

    def to_json(self):
        return dict(super().to_json(), **{
            'nc': [nc.to_json() for nc in self.notes]
        })

    def compute_basic_neume_type(self) -> BasicNeumeType:
        if len(self.notes) == 1:
            # No difference between virga and punktum
            return BasicNeumeType.VIRGA
        elif len(self.notes) == 2:
            if self.notes[1].graphical_connection == GraphicalConnectionType.GAPED:
                if self.notes[0].graphical_connection > self.notes[1].graphical_connection > self.notes[2].graphical_connection:
                    return BasicNeumeType.CLIMACUS
                return BasicNeumeType.DISTROPHA
            else:
                if self.notes[1].position_in_staff > self.notes[0].position_in_staff:
                    return BasicNeumeType.PES
                else:
                    return BasicNeumeType.CLIVIS
        elif len(self.notes) == 3:
            if self.notes[1].graphical_connection == GraphicalConnectionType.GAPED:
                if self.notes[2].graphical_connection == GraphicalConnectionType.GAPED:
                    if self.notes[0].position_in_staff > self.notes[1].position_in_staff > self.notes[2].position_in_staff:
                        return BasicNeumeType.CLIMACUS
                    return BasicNeumeType.TRISTROPHA
                else:
                    if self.notes[2].position_in_staff > self.notes[1].position_in_staff:
                        return BasicNeumeType.SCANDICUS
                    else:
                        return BasicNeumeType.PRESSUS
            else:
                if self.notes[2].graphical_connection == GraphicalConnectionType.GAPED:
                    if self.notes[1].position_in_staff > self.notes[0].position_in_staff:
                        return BasicNeumeType.SCANDICUS
                    else:
                        return BasicNeumeType.ORISCUS
                else:
                    if self.notes[2].position_in_staff > self.notes[1].position_in_staff > self.notes[0].position_in_staff:
                        return BasicNeumeType.SCANDICUS
                    elif self.notes[0].position_in_staff > self.notes[1].position_in_staff < self.notes[2].position_in_staff:
                        return BasicNeumeType.PORRECTUS
                    elif self.notes[2].position_in_staff < self.notes[1].position_in_staff > self.notes[0].position_in_staff:
                        return BasicNeumeType.TORCULUS
                    else:
                        return BasicNeumeType.OTHER
        elif len(self.notes) == 4:
            if self.notes[0].position_in_staff < self.notes[1].position_in_staff >= self.notes[2].position_in_staff >= self.notes[3].position_in_staff:
                if self.notes[1].graphical_connection == GraphicalConnectionType.LOOPED and self.notes[2].graphical_connection == GraphicalConnectionType.GAPED and self.notes[3].graphical_connection == GraphicalConnectionType.GAPED:
                    return BasicNeumeType.PODATUS_SUBBUPUNCTIS
            if self.notes[0].position_in_staff >= self.notes[1].position_in_staff >= self.notes[2].position_in_staff >= self.notes[3].position_in_staff:
                if self.notes[1].graphical_connection == GraphicalConnectionType.GAPED and self.notes[2].graphical_connection == GraphicalConnectionType.GAPED and self.notes[3].graphical_connection == GraphicalConnectionType.GAPED:
                    return BasicNeumeType.CLIMACUS
            return BasicNeumeType.OTHER
        elif len(self.notes) == 5:
            if self.notes[0].position_in_staff < self.notes[1].position_in_staff >= self.notes[2].position_in_staff >= self.notes[3].position_in_staff:
                if self.notes[1].graphical_connection == GraphicalConnectionType.LOOPED and self.notes[2].graphical_connection == GraphicalConnectionType.GAPED and self.notes[3].graphical_connection == GraphicalConnectionType.GAPED:
                    return BasicNeumeType.PODATUS_SUBBUPUNCTIS
            if self.notes[0].position_in_staff >= self.notes[1].position_in_staff >= self.notes[2].position_in_staff >= self.notes[3].position_in_staff:
                if self.notes[1].graphical_connection == GraphicalConnectionType.GAPED and self.notes[2].graphical_connection == GraphicalConnectionType.GAPED and self.notes[3].graphical_connection == GraphicalConnectionType.GAPED:
                    return BasicNeumeType.CLIMACUS
            return BasicNeumeType.OTHER
        else:
            return BasicNeumeType.OTHER


        raise Exception('Unknown')




class ClefType(IntEnum):
    CLEF_F = 0
    CLEF_C = 1


class Clef(Symbol):
    def __init__(self,
                 clef_type=ClefType.CLEF_F,
                 coord=Point(),
                 position_in_staff=MusicSymbolPositionInStaff.UNDEFINED,
                 fixed_sorting: bool = False,
                 s_id: str = None
                 ):
        super().__init__(s_id, SymbolType.CLEF, fixed_sorting, coord, position_in_staff)
        self.clef_type = clef_type

    @staticmethod
    def from_json(json: dict):
        return Clef(
            ClefType(json.get("type", ClefType.CLEF_F)),
            Point.from_json(json.get("coord", "0,0")),
            MusicSymbolPositionInStaff(json.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED)),
            json.get('fixedSorting', False),
            json.get('id', None)
        )

    def to_json(self):
        return dict(super().to_json(), **{
            "type": self.clef_type.value,
        })




class MusicLine:
    def __init__(self,
                 ml_id: str = None,
                 coords: Coords = None,
                 staff_lines: StaffLines = None,
                 symbols: List[Symbol] = None,
                 reconstructed=False,
                 ):
        self.id = ml_id if ml_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.staff_lines = staff_lines if staff_lines else StaffLines()
        self.symbols = symbols if symbols else []
        self.reconstructed = reconstructed
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
            json.get('reconstructed', False),
        )

    def to_json(self):
        return {
            "id": self.id,
            "coords": self.coords.to_json(),
            "staffLines": self.staff_lines.to_json(),
            "symbols": [s.to_json() for s in self.symbols],
            'reconstructed': self.reconstructed,
        }

    def avg_line_distance(self, default=-1):
        return self.staff_lines.avg_line_distance(default)

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
        mls = MusicLines([MusicLine.from_json(l) for l in json])
        current_clef = None
        for ml in mls:
            current_clef = ml.update_note_names(current_clef)
        return mls

    def to_json(self):
        return [l.to_json() for l in self]

    def draw(self, canvas, color=(0, 255, 0), line_thickness=1):
        for ml in self:
            ml.draw(canvas, color=color, thickness=line_thickness)

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
