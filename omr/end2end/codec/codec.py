"""The token vocabulary (codec) of the end-to-end networks.

A music symbol is a single token, e.g. ``(note|0|6|2)``, ``(clef|c|7)`` or ``(accid|flat|5)``.
Built-in tokens are produced by ``MusicSymbol.__repr__`` and must stay byte-identical, since
every trained model ships the codec it was trained with (``codec.txt``) and its embedding rows
are addressed through it.

A symbol of a class registered at runtime (``database.models.symbolclasses.SymbolClass``)
carries the class id in the sub-type slot, prefixed by ``CLASS_PREFIX``:
``(clef|@clef_x|7)``, ``(note|@quilisma|6|2)``, ``(other|@barline|5)``. Class ids only consist
of ``\\w`` characters (see ``restapi/views/symbolclasses.py``), so they never collide with the
token delimiters.
"""
from typing import Optional, List, Collection, Iterable

from database.file_formats.pcgts import MusicSymbol, ClefType, NoteType, GraphicalConnectionType, AccidType, \
    MusicSymbolPositionInStaff, SymbolType, NoteName
import database.file_formats.pcgts as ns_pcgts

CLASS_PREFIX = '@'

BASE_TOKENS = ["NEW_CHANT", "[", "]", "*", "(", ")", "[NEW_CHANT]", "[NewLine]", "NewLine", "[NEW_STROPHE]"]

# gaped connections are folded into NEUME_START by the codec; order as in the legacy generator
NOTE_CONNECTIONS = [GraphicalConnectionType.NEUME_START, GraphicalConnectionType.LOOPED]


def symbol_token(s: MusicSymbol, class_ids: Collection[str] = (), melody: bool = False) -> Optional[str]:
    """The codec token of `s`, None if the codec cannot express it.

    A symbol whose registered class is in `class_ids` gets its class token; any other
    symbol falls back to its built-in base token, like the pixel classifier's label lookup
    (`omr.imageoperations.symbol_label_set.main_index`). A symbol of type OTHER has no
    base token, so without its class it is not encodable.
    """
    if s.symbol_class and s.symbol_class in class_ids:
        parts = [s.symbol_type.value, CLASS_PREFIX + s.symbol_class]
        if melody:
            parts += [str(s.note_name.value), str(s.octave)]
        else:
            parts.append(str(s.position_in_staff.value))
        if s.symbol_type == SymbolType.NOTE:
            gc = s.graphical_connection
            parts.append(str((GraphicalConnectionType.NEUME_START if gc == GraphicalConnectionType.GAPED else gc).value))
        return "({})".format("|".join(parts))

    if s.symbol_type == SymbolType.OTHER:
        return None
    return s.__repr__(ignore_gaped=True, melody=melody)


def is_symbol_token(token: str) -> bool:
    return len(token) > 2 and token[0] == '(' and token[-1] == ')' and '|' in token


def _pitch_variants(melody: bool) -> Iterable[dict]:
    if melody:
        for name in NoteName:
            for octave in range(15):
                yield dict(note_name=name, octave=octave)
    else:
        for pis in MusicSymbolPositionInStaff:
            yield dict(position_in_staff=pis)


def codec_lines(unique_chars: Optional[List[str]] = None, melody: bool = False,
                class_specs: Iterable = ()) -> List[str]:
    """All tokens of a codec. `class_specs` are the registered classes
    (`SymbolLabelSpec` with a `class_id`) that get their own tokens."""
    lines = list(BASE_TOKENS)
    if unique_chars is not None:
        lines += unique_chars

    pitches = list(_pitch_variants(melody))
    for clef in ClefType:
        for p in pitches:
            lines.append(MusicSymbol(symbol_type=SymbolType.CLEF, clef_type=clef, **p).__repr__(melody=melody))
    for accid in AccidType:
        for p in pitches:
            lines.append(MusicSymbol(symbol_type=SymbolType.ACCID, accid_type=accid, **p).__repr__(melody=melody))
    for note_type in NoteType:
        for p in pitches:
            for gc in NOTE_CONNECTIONS:
                lines.append(MusicSymbol(symbol_type=SymbolType.NOTE, note_type=note_type, graphical_connection=gc,
                                         **p).__repr__(melody=melody))

    for spec in class_specs:
        for p in pitches:
            connections = NOTE_CONNECTIONS if spec.symbol_type == SymbolType.NOTE else [None]
            for gc in connections:
                s = MusicSymbol(symbol_type=spec.symbol_type, symbol_class=spec.class_id, **p)
                if gc is not None:
                    s.graphical_connection = gc
                lines.append(symbol_token(s, {spec.class_id}, melody=melody))

    # stable de-duplication: the tokenizer assigns indices in file order
    return list(dict.fromkeys(lines))


def merge_codec_lines(base: List[str], new: List[str]) -> List[str]:
    """`base` unchanged followed by the tokens of `new` it lacks, so that every token of a
    pretrained codec keeps its index when fine-tuning extends the vocabulary."""
    known = set(base)
    return list(base) + [t for t in new if t not in known]


def read_codec(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def write_codec(file_path: str, lines: List[str]):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def gen_codec(file_path="/tmp/codec.txt", unique_chars: List[str] = None, melody=False, class_specs: Iterable = ()):
    write_codec(file_path, codec_lines(unique_chars, melody, class_specs))


if __name__ == "__main__":
    import string
    from database import DatabaseBook

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
