from database.file_formats.pcgts.page.musicsymbol import MusicSymbol
from omr.imageoperations import ImageOperationData
from database.file_formats.pcgts.page import MusicSymbol, SymbolType, \
    GraphicalConnectionType, AccidType, ClefType, NoteType, \
    MusicSymbolPositionInStaff, StaffLines, create_clef, create_accid
from typing import List, Tuple, Union, Optional, Any, Dict

CodecType = Tuple[SymbolType, Union[NoteType, ClefType, AccidType], MusicSymbolPositionInStaff, Optional[GraphicalConnectionType]]

class CalamariSequence:
    codec: Dict[CodecType, str] = {}
    decodec: Dict[str, CodecType] = {}
    last_char = ord('!')

    @classmethod
    def encode(cls, c: CodecType) -> str:
        if c not in cls.codec:
            cls.codec[c] = chr(cls.last_char)
            cls.decodec[cls.codec[c]] = c
            cls.last_char += 1

        return cls.codec[c]

    @classmethod
    def decode(cls, s: str) -> CodecType:
        return cls.decodec[s]

    @classmethod
    def type_of_symbol(cls, ms: MusicSymbol) -> CodecType:
        if ms.symbol_type == SymbolType.NOTE:
            return ms.symbol_type, ms.note_type, ms.position_in_staff, ms.graphical_connection
        elif ms.symbol_type == SymbolType.ACCID:
            return ms.symbol_type, ms.accid_type, ms.position_in_staff, None
        elif ms.symbol_type == SymbolType.CLEF:
            return ms.symbol_type, ms.clef_type, ms.position_in_staff, None
        raise Exception()

    @staticmethod
    def pos_to_pos_in_staff(p: int) -> MusicSymbolPositionInStaff:
        return MusicSymbolPositionInStaff(p)

    @classmethod
    def to_symbol_types(cls, s: str, user_data=None) -> List[CodecType]:
        return list(map(cls.decode, s))

    @staticmethod
    def to_symbols(s_p: List[Tuple[str, float]], staff_lines: StaffLines) -> List[MusicSymbol]:
        s = "".join([char for char, _ in s_p])
        pos = [x for _, x in s_p]
        out = []
        neume: Optional[Neume] = None
        for symbol_type, sub_type, pis, neume_start, x in CalamariSequence.to_symbol_types(s, pos):
            coord = staff_lines.compute_coord_by_position_in_staff(x, pis)
            out.append(MusicSymbol(None,
                                   symbol_type=symbol_type,
                                   coord=coord,
                                   position_in_staff=pis,
                                   graphical_connection=GraphicalConnectionType.NEUME_START if neume_start else sub_type,))

        return out

    def __init__(self, symbols: List[MusicSymbol]):
        out = list(map(CalamariSequence.encode, map(CalamariSequence.type_of_symbol, symbols)))

        self.symbols = symbols
        self.calamari_str = "".join(out).strip()


class RegionLineMaskData:
    def __init__(self, op: ImageOperationData):
        self.operation = op
        self.line_image = op.images[1].image if len(op.images) > 1 else op.images[0].image
        self.region = op.images[0].image
        self.mask = op.images[2].image if len(op.images) > 2 else None

    def calamari_sequence(self):
        return CalamariSequence(self.operation.music_line.symbols)

