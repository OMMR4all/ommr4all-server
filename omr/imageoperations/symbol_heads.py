from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Tuple, Type, TYPE_CHECKING

import numpy as np

from database.file_formats.pcgts import Page, PageScaleReference, Line, MusicSymbol, ClefType, AccidType, \
    SymbolType, NoteType

if TYPE_CHECKING:
    from segmentation.settings import ColorMap


class AdditionalSymbolLabel(IntEnum):
    BACKGROUND = 0
    NORMAL = 1
    ORISCUS = 2
    APOSTROPHA = 3
    LIQUESCENT_FOLLOWING_U = 4
    LIQUESCENT_FOLLOWING_D = 5
    CLEF_C = 6
    CLEF_F = 7
    ACCID_NATURAL = 8
    ACCID_SHARP = 9
    ACCID_FLAT = 10

    def get_color(self):
        return {0: [255, 255, 255],
                1: [255, 0, 0],
                2: [255, 120, 120],
                3: [120, 0, 0],
                4: [120, 255, 120],
                5: [0, 255, 0],
                6: [0, 0, 255],
                7: [50, 50, 255],
                8: [0, 0, 120],
                9: [60, 120, 120],
                10: [120, 60, 120]
                }[self.value]

    def get_note_type(self):
        return {
            AdditionalSymbolLabel.NORMAL: NoteType.NORMAL,
            AdditionalSymbolLabel.ORISCUS: NoteType.ORISCUS,
            AdditionalSymbolLabel.APOSTROPHA: NoteType.APOSTROPHA,
            AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_U: NoteType.LIQUESCENT_FOLLOWING_U,
            AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_D: NoteType.LIQUESCENT_FOLLOWING_D
        }[self] if self.value in [1, 2, 3, 4, 5] else None


def draw_note_type_mask(ml: Line, img: np.ndarray, page: Page, scale: PageScaleReference):
    import cv2

    if len(ml.staff_lines) < 2:  # at least two staff lines required
        return None

    def p2i(p):
        return page.page_to_image_scale(p, scale)

    radius = max(1, p2i(ml.staff_lines[-1].center_y() - ml.staff_lines[0].center_y()) / len(ml.staff_lines) / 8)

    def set(coord, label: AdditionalSymbolLabel, dx=radius, dy=radius):
        coord = p2i(coord)
        cv2.circle(img, tuple(coord.p.round().astype(int)), int(radius * 2), color=label.value, thickness=-1)

    for s in ml.symbols:
        if s.symbol_type == SymbolType.NOTE:
            if s.note_type == NoteType.LIQUESCENT_FOLLOWING_U:
                set(s.coord, AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_U)
            elif s.note_type == NoteType.LIQUESCENT_FOLLOWING_D:
                set(s.coord, AdditionalSymbolLabel.LIQUESCENT_FOLLOWING_D)
            elif s.note_type == NoteType.APOSTROPHA:
                set(s.coord, AdditionalSymbolLabel.APOSTROPHA)
            elif s.note_type == NoteType.ORISCUS:
                set(s.coord, AdditionalSymbolLabel.ORISCUS)
            else:
                set(s.coord, AdditionalSymbolLabel.NORMAL)
        elif s.symbol_type == SymbolType.CLEF:
            if s.clef_type == ClefType.F:
                set(s.coord, AdditionalSymbolLabel.CLEF_F, dy=4 * radius)
            elif s.clef_type == ClefType.C:
                set(s.coord, AdditionalSymbolLabel.CLEF_C, dy=4 * radius)
            # clef types without a trainable label are ignored (background)
        elif s.symbol_type == SymbolType.ACCID:
            if s.accid_type == AccidType.NATURAL:
                set(s.coord, AdditionalSymbolLabel.ACCID_NATURAL)
            elif s.accid_type == AccidType.FLAT:
                set(s.coord, AdditionalSymbolLabel.ACCID_FLAT)
            elif s.accid_type == AccidType.SHARP:
                set(s.coord, AdditionalSymbolLabel.ACCID_SHARP)
            # accid types without a trainable label are ignored (background)

    return img


def apply_note_type_label(symbol: MusicSymbol, label: AdditionalSymbolLabel):
    if symbol.symbol_type == SymbolType.NOTE:
        note_type = label.get_note_type()
        symbol.note_type = note_type if note_type is not None else NoteType.NORMAL


@dataclass(frozen=True)
class SymbolHeadSpec:
    """One additional (optional) network head predicting an independent symbol attribute.

    The main head (SymbolLabel) is fixed for checkpoint compatibility; every further
    attribute gets its own entry in SYMBOL_DETECTION_HEADS. Head i corresponds to the
    dataset column `add_mask_{i}` and albumentations target `mask_head_{i}`.
    """
    name: str
    labels: Type[IntEnum]  # enum with get_color(); len(labels) = number of classes
    draw_mask: Callable[[Line, np.ndarray, Page, PageScaleReference], None]  # rasterize GT labels in-place
    apply_label: Callable[[MusicSymbol, IntEnum], None]  # set the decoded attribute on a predicted symbol

    def color_map(self) -> 'ColorMap':
        from segmentation.settings import ColorMap, ClassSpec
        return ColorMap([ClassSpec(label=i.value, name=i.name.lower(), color=i.get_color())
                         for i in self.labels])


NOTE_TYPE_HEAD = SymbolHeadSpec(
    name='note_types',
    labels=AdditionalSymbolLabel,
    draw_mask=draw_note_type_mask,
    apply_label=apply_note_type_label,
)

SYMBOL_DETECTION_HEADS: Tuple[SymbolHeadSpec, ...] = (NOTE_TYPE_HEAD,)


def head_color_maps() -> List['ColorMap']:
    return [h.color_map() for h in SYMBOL_DETECTION_HEADS]


def head_classes() -> List[int]:
    return [len(h.labels) for h in SYMBOL_DETECTION_HEADS]
