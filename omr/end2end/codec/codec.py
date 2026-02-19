from typing import Optional, List

from database import DatabaseBook
from database.file_formats.pcgts import MusicSymbol, ClefType, NoteType, GraphicalConnectionType, AccidType, \
    MusicSymbolPositionInStaff, SymbolType, NoteName
import database.file_formats.pcgts as ns_pcgts
from database.file_formats.pcgts import Rect


def gen_codec(file_path="/tmp/codec.txt", unique_chars: List[str] = None, melody=False):
    codecs = ["NEW_CHANT", "[", "]", "*", "(", ")", "[NEW_CHANT]", "[NewLine]", "NewLine"]

    if unique_chars is not None:
        codecs = codecs + unique_chars

    with open(file_path, "w") as f:

        if not melody:

            for i in ClefType:
                for t in MusicSymbolPositionInStaff:
                    print(MusicSymbol(symbol_type=SymbolType.CLEF, clef_type=i, position_in_staff=t).__repr__())
                    codecs.append(MusicSymbol(symbol_type=SymbolType.CLEF, clef_type=i, position_in_staff=t).__repr__())

            for i in AccidType:
                for t in MusicSymbolPositionInStaff:
                    print(MusicSymbol(symbol_type=SymbolType.ACCID, accid_type=i, position_in_staff=t).__repr__())
                    codecs.append(
                        MusicSymbol(symbol_type=SymbolType.ACCID, accid_type=i, position_in_staff=t).__repr__())
            for i in NoteType:
                for t in MusicSymbolPositionInStaff:
                    for z in GraphicalConnectionType:
                        if z == z.GAPED:
                            z = z.NEUME_START
                        print(MusicSymbol(symbol_type=SymbolType.NOTE, note_type=i, position_in_staff=t,
                                          graphical_connection=z).__repr__())
                        codecs.append(MusicSymbol(symbol_type=SymbolType.NOTE, note_type=i, position_in_staff=t,
                                                  graphical_connection=z).__repr__())
        else:
            for i in ClefType:
                for t in NoteName:
                    for z in range(15):
                        print(MusicSymbol(symbol_type=SymbolType.CLEF, clef_type=i, note_name=t, octave=z).__repr__(
                            melody=melody))
                        codecs.append(
                            MusicSymbol(symbol_type=SymbolType.CLEF, clef_type=i, note_name=t, octave=z).__repr__(
                                melody=melody))

            for i in AccidType:
                for t in NoteName:
                    for z in range(15):
                        print(MusicSymbol(symbol_type=SymbolType.ACCID, accid_type=i, note_name=t, octave=z).__repr__(
                            melody=melody))
                        codecs.append(
                            MusicSymbol(symbol_type=SymbolType.ACCID, accid_type=i, note_name=t, octave=z).__repr__(
                                melody=melody))
            for i in NoteType:
                for t in NoteName:
                    for z in range(15):
                        for e in GraphicalConnectionType:
                            if e == e.GAPED:
                                e = e.NEUME_START
                            print(MusicSymbol(symbol_type=SymbolType.NOTE, note_type=i, note_name=t, octave=z,
                                              graphical_connection=e).__repr__(
                                melody=melody))
                            codecs.append(
                                MusicSymbol(symbol_type=SymbolType.NOTE, note_type=i, note_name=t, octave=z,
                                            graphical_connection=e).__repr__(
                                    melody=melody))
        for line in codecs:
            f.write(f"{line}\n")


if __name__ == "__main__":
    import string

    b = DatabaseBook("Geesebook1_complete_fixed_ro")
    c = DatabaseBook("mul_2_rsync_gt2")
    d = DatabaseBook("Koeln_Dombibl_1001b_part_gt")
    e = DatabaseBook("Pa_14819_gt")
    f = DatabaseBook("Geesebook2_andreas1")
    g = DatabaseBook("Graduel_Part_1_gt")
    h = DatabaseBook("Graduel_Part_2_gt")
    i = DatabaseBook("Graduel_Part_3_gt")

    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    numbers = string.digits
    sentence = []

    codec_chars = lowercase + uppercase + numbers
    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts')) for y in [c, d, e, f] for x in y.pages() if
             x.page_progress().verified_allowed()]
    sentence.append(codec_chars)

    for i in pcgts:
        lines = i.page.all_text_lines()
        for t in lines:
            sentence.append(t.text())
    unique_chars = sorted(list(set("".join(sentence))))
    gen_codec(unique_chars=unique_chars, melody=False)
