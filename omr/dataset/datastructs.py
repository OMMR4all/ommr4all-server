from omr.imageoperations import ImageOperationData
from database.file_formats.pcgts.page import MusicSymbol, SymbolType, \
    GraphicalConnectionType, AccidType, ClefType, NoteType, \
    MusicSymbolPositionInStaff, StaffLines, create_clef, create_accid
from typing import List, Tuple, Union, Optional, Any


class CalamariSequence:
    neume_types = "QQWERTYUTYU__ASIOPD"

    neume_start = " "
    pos_to_char =      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    clef_pos_to_char = [
        "`1234567890=",   # F-Clef
        "~!@#$%^&*()+",   # C-Clef
    ]
    accid_type_to_char = {AccidType.FLAT: '}', AccidType.NATURAL: '[', AccidType.SHARP: '['}
    char_to_accid_type = dict((value, key) for key, value in accid_type_to_char.items())

    @staticmethod
    def pos_to_pos_in_staff(p: int) -> MusicSymbolPositionInStaff:
        return MusicSymbolPositionInStaff(p)

    @staticmethod
    def to_symbol_types(s: str, user_data=None) -> List[Tuple[SymbolType, Union[AccidType, ClefType, GraphicalConnectionType],
                                              MusicSymbolPositionInStaff, int, Any]]:
        bool_to_graphical_connection_type = [GraphicalConnectionType.LOOPED, GraphicalConnectionType.GAPED]
        out = []
        neume_start = False
        if user_data is None:
            user_data = [] * len(out)
        for char, ud in zip(s, user_data):
            if char in CalamariSequence.char_to_accid_type:
                out.append((SymbolType.ACCID, CalamariSequence.char_to_accid_type[char], MusicSymbolPositionInStaff(0), 0, ud))
            elif char in CalamariSequence.clef_pos_to_char[ClefType.CLEF_C]:
                p = CalamariSequence.clef_pos_to_char[ClefType.CLEF_C].index(char)
                out.append((SymbolType.CLEF, ClefType.CLEF_C, CalamariSequence.pos_to_pos_in_staff(p), 0, ud))
            elif char in CalamariSequence.clef_pos_to_char[ClefType.CLEF_F]:
                p = CalamariSequence.clef_pos_to_char[ClefType.CLEF_F].index(char)
                out.append((SymbolType.CLEF, ClefType.CLEF_F, CalamariSequence.pos_to_pos_in_staff(p), 0, ud))
            else:
                if char == CalamariSequence.neume_start:
                    neume_start = True
                elif char in CalamariSequence.pos_to_char or char in CalamariSequence.pos_to_char.lower():
                    p = CalamariSequence.pos_to_char.index(char.upper())
                    out.append((SymbolType.NOTE_COMPONENT, bool_to_graphical_connection_type[char.islower()],
                                CalamariSequence.pos_to_pos_in_staff(p),
                                neume_start, ud))
                    neume_start = False
                else:
                    print(CalamariSequence.char_to_accid_type)
                    raise ValueError('Unknown char {} in sequence {}'.format(char, s))

        return out

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
        out = []
        nt = []
        last_was_neume = False
        for s in symbols:
            if s.symbol_type == SymbolType.NOTE:
                if len(out) > 0 and out[-1] != ' ' and last_was_neume:
                    out.append(CalamariSequence.neume_start)

                last_was_neume = True

                n: Neume = s
                for i, nc in enumerate(n.notes):
                    char = CalamariSequence.pos_to_char[nc.position_in_staff.value]
                    if i > 0:
                        if nc.graphical_connection == GraphicalConnectionType.LOOPED:
                            char = char.upper()
                        elif nc.graphical_connection == GraphicalConnectionType.GAPED:
                            char = char.lower()
                        else:
                            raise ValueError("Unknown connection type {}".format(nc.graphical_connection))

                    out.append(char)
                nt.append(CalamariSequence.neume_types[n.compute_basic_neume_type()])
            elif s.symbol_type == SymbolType.CLEF:
                last_was_neume = False
                out.append(CalamariSequence.clef_pos_to_char[s.clef_type.value][s.position_in_staff.value])
                nt.append(CalamariSequence.clef_pos_to_char[s.clef_type.value][0])
            elif s.symbol_type == SymbolType.ACCID:
                last_was_neume = False
                out.append(CalamariSequence.accid_type_to_char[s.accid_type.value])
                nt.append(CalamariSequence.accid_type_to_char[s.accid_type.value])

        self.symbols = symbols
        self.calamari_str = "".join(out).strip()
        self.calamari_neume_types_str = "".join(nt)


class RegionLineMaskData:
    def __init__(self, op: ImageOperationData):
        self.operation = op
        self.line_image = op.images[1].image
        self.region = op.images[0].image
        self.mask = op.images[2].image if len(op.images) > 2 else None

    def calamari_sequence(self):
        return CalamariSequence(self.operation.music_line.symbols)

