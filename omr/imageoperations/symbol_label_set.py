"""The label inventory of the symbol-detection networks.

The built-in labels reproduce the two frozen enums (`SymbolLabel` for the main head,
`AdditionalSymbolLabel` for the note-type head) byte-for-byte and always occupy the
leading indices, so an already-trained checkpoint stays index-compatible. Classes
registered at runtime (`database.models.symbolclasses.SymbolClass`) are appended.

This module must only import `database.file_formats.pcgts` and the two label enums at
module level, so that both `omr/imageoperations/*` and `omr/dataset/*` can import it
without a cycle. The Django query is a function-local import.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from mashumaro.mixins.json import DataClassJSONMixin

from database.file_formats.pcgts import MusicSymbol, SymbolType, ClefType, AccidType, NoteType, \
    GraphicalConnectionType
from omr.imageoperations.symbol_heads import AdditionalSymbolLabel

import logging

if TYPE_CHECKING:
    from segmentation.settings import ColorMap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolLabelSpec(DataClassJSONMixin):
    """One class of a symbol-detection network head.

    `sub_type` is the wire value of the attribute the label encodes; its meaning
    depends on the head and on `symbol_type`:
      main head, NOTE   -> GraphicalConnectionType value as str ('0' | '1' | '2')
      main head, CLEF   -> ClefType value ('c' | 'f' | 'g' | ...)
      main head, ACCID  -> AccidType value ('flat' | 'natural' | 'sharp' | ...)
      note-type head    -> NoteType value as str ('0' .. '4')
    `symbol_type` is None for the background label (index 0). `class_id` is the
    SymbolClass.id for a registered class and None for a built-in label.
    """
    index: int
    id: str
    color: List[int]
    symbol_type: Optional[SymbolType] = None
    sub_type: str = ''
    class_id: Optional[str] = None


def builtin_main_labels() -> List[SymbolLabelSpec]:
    from omr.imageoperations.music_line_operations import SymbolLabel
    base = {
        SymbolLabel.BACKGROUND: (None, ''),
        SymbolLabel.NOTE_START: (SymbolType.NOTE, str(GraphicalConnectionType.NEUME_START.value)),
        SymbolLabel.NOTE_LOOPED: (SymbolType.NOTE, str(GraphicalConnectionType.LOOPED.value)),
        SymbolLabel.NOTE_GAPPED: (SymbolType.NOTE, str(GraphicalConnectionType.GAPED.value)),
        SymbolLabel.CLEF_C: (SymbolType.CLEF, ClefType.C.value),
        SymbolLabel.CLEF_F: (SymbolType.CLEF, ClefType.F.value),
        SymbolLabel.ACCID_NATURAL: (SymbolType.ACCID, AccidType.NATURAL.value),
        SymbolLabel.ACCID_SHARP: (SymbolType.ACCID, AccidType.SHARP.value),
        SymbolLabel.ACCID_FLAT: (SymbolType.ACCID, AccidType.FLAT.value),
    }
    return [SymbolLabelSpec(index=l.value, id=l.name.lower(), color=l.get_color(),
                            symbol_type=base[l][0], sub_type=base[l][1])
            for l in SymbolLabel]


_ADDITIONAL_BASE = {
    AdditionalSymbolLabel.BACKGROUND: (None, ''),
    AdditionalSymbolLabel.NORMAL: (SymbolType.NOTE, str(NoteType.NORMAL.value)),
    AdditionalSymbolLabel.ORISCUS: (SymbolType.NOTE, str(NoteType.ORISCUS.value)),
    AdditionalSymbolLabel.APOSTROPHA: (SymbolType.NOTE, str(NoteType.APOSTROPHA.value)),
    AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_U: (SymbolType.NOTE,
                                                   str(NoteType.LIQUESCENT_FOLLOWING_U.value)),
    AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_D: (SymbolType.NOTE,
                                                   str(NoteType.LIQUESCENT_FOLLOWING_D.value)),
    AdditionalSymbolLabel.CLEF_C: (SymbolType.CLEF, ClefType.C.value),
    AdditionalSymbolLabel.CLEF_F: (SymbolType.CLEF, ClefType.F.value),
    AdditionalSymbolLabel.ACCID_NATURAL: (SymbolType.ACCID, AccidType.NATURAL.value),
    AdditionalSymbolLabel.ACCID_SHARP: (SymbolType.ACCID, AccidType.SHARP.value),
    AdditionalSymbolLabel.ACCID_FLAT: (SymbolType.ACCID, AccidType.FLAT.value),
}


def builtin_note_type_labels() -> List[SymbolLabelSpec]:
    return [SymbolLabelSpec(index=l.value, id=l.name.lower(), color=l.get_color(),
                            symbol_type=_ADDITIONAL_BASE[l][0], sub_type=_ADDITIONAL_BASE[l][1])
            for l in AdditionalSymbolLabel]


def _builtin_colors():
    from omr.imageoperations.music_line_operations import SymbolLabel
    return {tuple(l.get_color()) for l in SymbolLabel} | \
           {tuple(l.get_color()) for l in AdditionalSymbolLabel}


def color_for_custom_label(n: int) -> List[int]:
    """n-th colour of the 6x6x6 web-safe cube walked downwards, skipping built-in colours."""
    builtin = _builtin_colors()
    remaining, v = n, 215
    while v >= 0:
        c = ((v // 36) * 51, ((v // 6) % 6) * 51, (v % 6) * 51)
        if c not in builtin:
            if remaining == 0:
                return list(c)
            remaining -= 1
        v -= 1
    raise ValueError('too many symbol classes for the mask colour space')


def _color_map(labels: List[SymbolLabelSpec]) -> 'ColorMap':
    from segmentation.settings import ColorMap, ClassSpec
    return ColorMap([ClassSpec(label=s.index, name=s.id, color=s.color) for s in labels])


def note_type_index(s: MusicSymbol, labels: List[SymbolLabelSpec]) -> int:
    """Index of `s` in a note-type head label list, 0 (background) if it has none."""
    if s.symbol_class:
        for spec in labels:
            if spec.class_id == s.symbol_class:
                return spec.index
    if s.symbol_type == SymbolType.NOTE:
        sub_type = str(s.note_type.value)
    elif s.symbol_type == SymbolType.CLEF:
        sub_type = s.clef_type.value
    elif s.symbol_type == SymbolType.ACCID:
        sub_type = s.accid_type.value
    else:
        return 0
    for spec in labels:
        if spec.class_id is None and spec.symbol_type == s.symbol_type and spec.sub_type == sub_type:
            return spec.index
    return 0


def main_index(s: MusicSymbol, labels: List[SymbolLabelSpec], keep_graphical_connection=None) -> int:
    """Index of `s` in a main head label list, 0 (background) if it has none."""
    if s.symbol_class:
        for spec in labels:
            if spec.class_id == s.symbol_class:
                return spec.index

    if s.symbol_type == SymbolType.NOTE:
        if keep_graphical_connection and len(keep_graphical_connection) == 3:
            if keep_graphical_connection[1] and s.graphical_connection == GraphicalConnectionType.GAPED:
                connection = GraphicalConnectionType.GAPED
            elif keep_graphical_connection[2] and s.graphical_connection == GraphicalConnectionType.LOOPED:
                connection = GraphicalConnectionType.LOOPED
            else:
                connection = GraphicalConnectionType.NEUME_START
        else:
            connection = s.graphical_connection
        sub_type = str(connection.value)
    elif s.symbol_type == SymbolType.CLEF:
        sub_type = s.clef_type.value
    elif s.symbol_type == SymbolType.ACCID:
        sub_type = s.accid_type.value
    else:
        logger.warning('Symbol type {} has no trainable label, treating as background'.format(s.symbol_type))
        return 0

    for spec in labels:
        if spec.class_id is None and spec.symbol_type == s.symbol_type and spec.sub_type == sub_type:
            return spec.index

    if s.symbol_type == SymbolType.CLEF:
        logger.warning('Clef type {} has no trainable label, treating as background'.format(s.clef_type))
    elif s.symbol_type == SymbolType.ACCID:
        logger.warning('Accid type {} has no trainable label, treating as background'.format(s.accid_type))
    return 0


@dataclass
class SymbolClassLabelSets(DataClassJSONMixin):
    main: List[SymbolLabelSpec] = field(default_factory=list)
    note_type: List[SymbolLabelSpec] = field(default_factory=list)

    @staticmethod
    def builtin() -> 'SymbolClassLabelSets':
        return SymbolClassLabelSets(builtin_main_labels(), builtin_note_type_labels())

    def main_index_of(self, s: MusicSymbol, keep_graphical_connection=None) -> int:
        return main_index(s, self.main, keep_graphical_connection)

    def note_type_index_of(self, s: MusicSymbol) -> int:
        return note_type_index(s, self.note_type)

    def main_color_map(self) -> 'ColorMap':
        return _color_map(self.main)

    def note_type_color_map(self) -> 'ColorMap':
        return _color_map(self.note_type)


def symbol_label_sets_for_style(style_id: Optional[str]) -> SymbolClassLabelSets:
    """The label inventory a book of notation style `style_id` trains and predicts with."""
    from django.db.models import Q
    from database.models.symbolclasses import SymbolClass

    sets = SymbolClassLabelSets.builtin()
    rows = SymbolClass.objects.filter(Q(style_id=style_id) | Q(style__isnull=True))
    n_custom = {'main': 0, 'note_type': 0}
    for row in rows:
        if row.base_symbol_type in ('clef', 'accid'):
            target, key = sets.main, 'main'
        elif row.base_symbol_type == 'note':
            target, key = sets.note_type, 'note_type'
        else:
            logger.warning('Symbol class {} has an unknown base symbol type {}, skipping'.format(
                row.id, row.base_symbol_type))
            continue
        target.append(SymbolLabelSpec(
            index=len(target),
            id=row.id,
            color=color_for_custom_label(n_custom[key]),
            symbol_type=SymbolType(row.base_symbol_type),
            sub_type=row.base_sub_type,
            class_id=row.id,
        ))
        n_custom[key] += 1
    return sets
