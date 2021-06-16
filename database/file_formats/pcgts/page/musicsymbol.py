from typing import Optional, TYPE_CHECKING
from uuid import uuid4
from .coords import Point
from .definitions import MusicSymbolPositionInStaff
from enum import IntEnum, Enum
from shared.jsonparsing import optional_enum

if TYPE_CHECKING:
    from .staffline import StaffLines


class SymbolType(Enum):
    NOTE = 'note'
    CLEF = 'clef'
    ACCID = 'accid'


class NoteType(IntEnum):
    NORMAL = 0
    ORISCUS = 1
    APOSTROPHA = 2
    LIQUESCENT_FOLLOWING_U = 3
    LIQUESCENT_FOLLOWING_D = 4


class GraphicalConnectionType(IntEnum):
    GAPED = 0
    LOOPED = 1
    NEUME_START = 2


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

    @staticmethod
    def from_string(name: str):
        for name, member in NoteName.__members__.items():
            if name.lower() == name.lower():
                return member
        return NoteName.UNDEFINED


# See https://music-encoding.org/guidelines/v3/content/neumes.html (Fig. 1) for reference
class BasicNeumeType(IntEnum):
    # Single Notes
    VIRGA = 0
    PUNCTA = 1

    # Two-note neumes
    PES = 2
    CLIVIS = 3

    # three-note neumes
    SCANDICUS = 4
    CLIMACUS = 5
    TORCULUS = 6
    PORRECTUS = 7

    # four-note neumes
    PODATUS_SUBBUPUNCTIS = 8
    TORCULUS_RESUPINUS = 9
    PORRECTUS_FLEXUS = 10

    # Liquescent neumes
    EPIPHONUS = 11
    CEPHALICUS = 12

    # Stropic neumes
    DISTROPHA = 13
    TRISTROPHA = 14
    ORISCUS = 15
    PRESSUS = 16

    # Special neumes
    SALICUS = 17
    QUILISMA = 18

    OTHER = -1


class ClefType(Enum):
    C = 'c'
    F = 'f'

    def offset(self):
        return {
            ClefType.C: -3,
            ClefType.F: 0,
        }[self]


class AccidType(Enum):
    FLAT = 'flat'
    NATURAL = 'natural'
    SHARP = 'sharp'


class SymbolPredictionConfidence:
    def __init__(self, background: float = None, note_start=None, note_looped=None, note_gapped=None,
                 clef_c=None, clef_f=None, accid_natural=None, accid_sharp=None, accid_flat=None):
        self.background = background
        self.note_start = note_start
        self.note_looped = note_looped
        self.note_gapped = note_gapped
        self.clef_c = clef_c
        self.clef_f = clef_f
        self.accid_natural = accid_natural
        self.accid_sharp = accid_sharp
        self.accid_flat = accid_flat

    @staticmethod
    def from_json(d: dict):
        return SymbolPredictionConfidence(
            d.get('background', None),
            d.get('noteStart', None),
            d.get('noteLooped', None),
            d.get('noteGapped', None),
            d.get('clefC', None),
            d.get('clefF', None),
            d.get('accidNatural', None),
            d.get('accidSharp', None),
            d.get('accidFlat', None)
        ) if d else SymbolPredictionConfidence()

    def to_json(self):
        return {
            'background': self.background,
            'noteStart': self.note_start,
            'noteLooped': self.note_looped,
            'noteGapped': self.note_gapped,
            'clefC': self.clef_c,
            'clefF': self.clef_f,
            'accidNatural': self.accid_natural,
            'accidSharp': self.accid_sharp,
            'accidFlat': self.accid_flat,
        }


class SymbolSequenceConfidence:
    def __init__(self, confidence: float = None, token_length: int = None):
        self.confidence: float = confidence
        self.token_length: int = token_length

    @staticmethod
    def from_json(d: dict):
        return SymbolSequenceConfidence(
            d.get('confidence', None),
            d.get('tokenLength', None),
        ) if d else SymbolSequenceConfidence()

    def to_json(self):
        return {
            'confidence': self.confidence,
            'tokenLength': self.token_length,
        }


class SymbolConfidence:
    def __init__(self,
                 symbol_prediction_confidence: SymbolPredictionConfidence = None,
                 symbol_sequence_confidence: SymbolSequenceConfidence = None,
                 ):
        self.symbol_prediction_confidence = symbol_prediction_confidence
        self.symbol_sequence_confidence = symbol_sequence_confidence

    @staticmethod
    def from_json(d: dict) -> 'SymbolConfidence':
        return SymbolConfidence(
            SymbolPredictionConfidence.from_json(d.get('symbolPredictionConfidence', None)),
            SymbolSequenceConfidence.from_json(d.get('symbolSequenceConfidence', None)),
        ) if d else SymbolConfidence()

    def to_json(self) -> dict:
        return {
            'symbolPredictionConfidence': self.symbol_prediction_confidence.to_json() if self.symbol_prediction_confidence else None,
            'symbolSequenceConfidence': self.symbol_sequence_confidence.to_json() if self.symbol_sequence_confidence else None}


class MusicSymbol:
    def __init__(self,
                 symbol_type: SymbolType,
                 s_id: Optional[str] = None,
                 note_type: Optional[NoteType] = None,
                 clef_type: Optional[ClefType] = None,
                 accid_type: Optional[AccidType] = None,
                 fixed_sorting: bool = False,
                 coord: Point = None,
                 position_in_staff: MusicSymbolPositionInStaff = MusicSymbolPositionInStaff.UNDEFINED,
                 octave: int = 0,
                 note_name: NoteName = NoteName.UNDEFINED,
                 graphical_connection: GraphicalConnectionType = GraphicalConnectionType.GAPED,
                 confidence: SymbolConfidence = None
                 ):
        self.id = s_id if s_id else str(uuid4())
        self.coord = coord if coord else Point()
        self.symbol_type = symbol_type
        self.note_type = note_type if note_type else NoteType.NORMAL
        self.clef_type = clef_type if clef_type else ClefType.C
        self.accid_type = accid_type if accid_type else AccidType.NATURAL
        self.fixed_sorting = fixed_sorting
        self.position_in_staff = position_in_staff
        self.octave = octave
        self.note_name = note_name
        self.graphical_connection = graphical_connection
        self.symbol_confidence = confidence

    @staticmethod
    def from_json(d: dict) -> 'MusicSymbol':
        if d.get('accidType') == 'neutral':
            del d['accidType']
        return MusicSymbol(
            SymbolType(d.get('type')),
            d.get('id', None),
            optional_enum(d, 'noteType', NoteType, None),
            optional_enum(d, 'clefType', ClefType, None),
            optional_enum(d, 'accidType', AccidType, None),
            d.get('fixedSorting', False),
            Point.from_json(d.get('coord', '')),
            MusicSymbolPositionInStaff(d.get('positionInStaff', MusicSymbolPositionInStaff.UNDEFINED.value)),
            d.get('oct', -1),
            optional_enum(d, 'pname', NoteName, NoteName.UNDEFINED),
            optional_enum(d, 'graphicalConnection', GraphicalConnectionType, GraphicalConnectionType.GAPED),
            SymbolConfidence.from_json(d.get('symbolConfidence', None))
        )

    def to_json(self) -> dict:
        # export to dict, but omit non required fields for type
        d = {
            'id': self.id,
            'type': self.symbol_type.value,
            'fixedSorting': self.fixed_sorting,
            'coord': self.coord.to_json(),
            'positionInStaff': self.position_in_staff.value,
        }
        if self.symbol_type == SymbolType.NOTE:
            d['noteType'] = self.note_type.value
            d['oct'] = self.octave
            d['pname'] = self.note_name.value
            d['graphicalConnection'] = self.graphical_connection.value
        elif self.symbol_type == SymbolType.CLEF:
            d['clefType'] = self.clef_type.value
        elif self.symbol_type == SymbolType.ACCID:
            d['accidType'] = self.accid_type.value

        d['symbolConfidence'] = self.symbol_confidence.to_json() if self.symbol_confidence else None
        return d

    def update_note_name(self, clef, staff_lines: Optional['StaffLines'] = None):
        if self.position_in_staff == MusicSymbolPositionInStaff.UNDEFINED:
            if staff_lines:
                self.position_in_staff = staff_lines.position_in_staff(self.coord)

        self.note_name, self.octave = clef.note_name_octave(self.position_in_staff)

    def note_name_octave(self, position_in_staff: MusicSymbolPositionInStaff):
        if not self.symbol_type == SymbolType.CLEF:
            raise TypeError("Expected type {} but has {}".format(SymbolType.CLEF, self.symbol_type))

        clef_type_offset = self.clef_type.offset()
        relative_offset = (position_in_staff - self.position_in_staff)
        note_name = NoteName((clef_type_offset + 49 + MusicSymbolPositionInStaff.LINE_2 + relative_offset) % 7)
        octave = 4 + (clef_type_offset + MusicSymbolPositionInStaff.LINE_1 + relative_offset) // 7
        octave = octave - 1 if self.clef_type == self.clef_type.F else octave
        return note_name, octave


def create_clef(
        clef_type: ClefType,
        s_id: Optional[str] = None,
        coord: Point = None,
        position_in_staff: MusicSymbolPositionInStaff = MusicSymbolPositionInStaff.UNDEFINED,
        confidence=None
):
    return MusicSymbol(
        SymbolType.CLEF,
        s_id,
        clef_type=clef_type,
        coord=coord,
        position_in_staff=position_in_staff,
        confidence=confidence

    )


def create_accid(
        accid_type: AccidType,
        s_id: Optional[str] = None,
        coord: Point = None,
        position_in_staff: MusicSymbolPositionInStaff = MusicSymbolPositionInStaff.UNDEFINED,
        confidence=None

):
    return MusicSymbol(
        SymbolType.ACCID,
        s_id,
        accid_type=accid_type,
        coord=coord,
        position_in_staff=position_in_staff,
        confidence=confidence

    )
