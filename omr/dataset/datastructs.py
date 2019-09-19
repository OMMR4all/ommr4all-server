from database.file_formats.pcgts.page.musicsymbol import MusicSymbol
from omr.imageoperations import ImageOperationData
from database.file_formats.pcgts.page import MusicSymbol, SymbolType, \
    GraphicalConnectionType, AccidType, ClefType, NoteType, \
    MusicSymbolPositionInStaff, StaffLines, create_clef, create_accid
from typing import List, Tuple, Union, Optional, Any, Dict, NamedTuple
from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin

from mashumaro.types import SerializableType


@dataclass(frozen=True)
class CodecType(DataClassJSONMixin):
    symbol_type: SymbolType
    note_type: Optional[NoteType] = None
    clef_type: Optional[ClefType] = None
    accid_type: Optional[AccidType] = None
    pos_in_staff: Optional[MusicSymbolPositionInStaff] = None
    graphical_connection: Optional[GraphicalConnectionType] = None


@dataclass
class CalamariCodec(SerializableType):
    codec: Dict[CodecType, str] = field(default_factory=lambda: {})
    decodec: Dict[str, CodecType] = field(default_factory=lambda: {})
    last_char: int = ord('!')

    def _serialize(self):
        return {
            #'decodec': self.decodec,
            'decodec': {k: v.to_dict() for k, v in self.decodec.items()},
            'last_char': self.last_char,
        }

    @classmethod
    def _deserialize(cls, value):
        codec = {}
        decodec = {}
        for k, v in value.get('decodec', {}).items():
            v = CodecType.from_dict(v)
            codec[v] = k
            decodec[k] = v
        return CalamariCodec(
            codec,
            decodec,
            value.get('last_char', ord('!')),
        )

    def to_dict(self, *args, **kwargs):
        return self._serialize()

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        return cls._deserialize(d)

    def encode(self, c: CodecType) -> str:
        if c not in self.codec:
            self.codec[c] = chr(self.last_char)
            self.decodec[self.codec[c]] = c
            self.last_char += 1

        return self.codec[c]

    def decode(self, s: str) -> CodecType:
        return self.decodec[s]

    @staticmethod
    def type_of_symbol(ms: MusicSymbol) -> CodecType:
        if ms.symbol_type == SymbolType.NOTE:
            return CodecType(ms.symbol_type, ms.note_type, None, None, ms.position_in_staff, ms.graphical_connection)
        elif ms.symbol_type == SymbolType.ACCID:
            return CodecType(ms.symbol_type, None, None, ms.accid_type, ms.position_in_staff, None)
        elif ms.symbol_type == SymbolType.CLEF:
            return CodecType(ms.symbol_type, None, ms.clef_type, None, ms.position_in_staff, None)
        raise Exception()

    @staticmethod
    def pos_to_pos_in_staff(p: int) -> MusicSymbolPositionInStaff:
        return MusicSymbolPositionInStaff(p)

    @classmethod
    def to_symbol_types(self, s: str) -> List[CodecType]:
        return list(map(self.decode, s))


class CalamariSequence:
    @staticmethod
    def to_symbols(codec: CalamariCodec, s_p: List[Tuple[str, float]], staff_lines: StaffLines) -> List[MusicSymbol]:
        out = []
        for s, pos in s_p:
            c = codec.decode(s)
            coord = staff_lines.compute_coord_by_position_in_staff(pos, c.pos_in_staff)
            out.append(MusicSymbol(symbol_type=c.symbol_type,
                                   clef_type=c.clef_type,
                                   note_type=c.note_type,
                                   accid_type=c.accid_type,
                                   coord=coord,
                                   position_in_staff=c.pos_in_staff,
                                   graphical_connection=c.graphical_connection,
                                   ))

        return out

    def __init__(self, codec: CalamariCodec, symbols: List[MusicSymbol]):
        out = list(map(codec.encode, map(codec.type_of_symbol, symbols)))

        self.codec = codec
        self.symbols = symbols
        self.calamari_str = "".join(out).strip()


class RegionLineMaskData:
    def __init__(self, op: ImageOperationData):
        self.operation = op
        self.line_image = op.images[1].image if len(op.images) > 1 else op.images[0].image
        self.region = op.images[0].image
        self.mask = op.images[2].image if len(op.images) > 2 else None

    def calamari_sequence(self, codec: CalamariCodec):
        return CalamariSequence(codec, self.operation.music_line.symbols)


if __name__ == "__main__":
    @dataclass
    class Test(DataClassJSONMixin):
        codec: CalamariCodec

    codec = CalamariCodec()
    codec.encode(CodecType(SymbolType.NOTE, NoteType.APOSTROPHA, None, None, MusicSymbolPositionInStaff.LINE_4, GraphicalConnectionType.NEUME_START))
    print(codec.decodec)
    print(codec.codec)
    enco = Test(codec).to_json()
    deco = Test.from_json(enco)
    print(Test(codec).to_json())