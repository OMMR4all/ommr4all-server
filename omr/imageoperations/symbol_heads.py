from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Tuple, TYPE_CHECKING

import numpy as np

from database.file_formats.pcgts import Page, PageScaleReference, Line, MusicSymbol, ClefType, AccidType, \
    SymbolType, NoteType

if TYPE_CHECKING:
    from segmentation.settings import ColorMap
    from omr.imageoperations.symbol_label_set import SymbolLabelSpec, SymbolClassLabelSets


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


def draw_note_type_mask(ml: Line, img: np.ndarray, page: Page, scale: PageScaleReference,
                        labels: List['SymbolLabelSpec']):
    import cv2
    from omr.imageoperations.symbol_label_set import note_type_index

    if len(ml.staff_lines) < 2:  # at least two staff lines required
        return None

    def p2i(p):
        return page.page_to_image_scale(p, scale)

    radius = max(1, p2i(ml.staff_lines[-1].center_y() - ml.staff_lines[0].center_y()) / len(ml.staff_lines) / 8)

    for s in ml.symbols:
        index = note_type_index(s, labels)
        if index:
            coord = p2i(s.coord)
            cv2.circle(img, tuple(coord.p.round().astype(int)), int(radius * 2), color=index, thickness=-1)

    return img


def apply_note_type_label(symbol: MusicSymbol, spec: 'SymbolLabelSpec'):
    if symbol.symbol_type != SymbolType.NOTE:
        return
    symbol.note_type = NoteType(int(spec.sub_type)) if spec.sub_type else NoteType.NORMAL
    if spec.class_id:
        symbol.symbol_class = spec.class_id


@dataclass(frozen=True)
class SymbolHeadSpec:
    """One additional (optional) network head predicting an independent symbol attribute.

    The main head (`SymbolClassLabelSets.main`) is the primary label inventory; every
    further attribute gets its own entry in `symbol_detection_heads()`. Head i
    corresponds to the dataset column `add_mask_{i}` and albumentations target
    `mask_head_{i}`.
    """
    name: str
    labels: List['SymbolLabelSpec']  # one spec per class; len(labels) = number of classes
    # rasterize GT labels in-place
    draw_mask: Callable[[Line, np.ndarray, Page, PageScaleReference, List['SymbolLabelSpec']], None]
    # set the decoded attribute on a predicted symbol
    apply_label: Callable[[MusicSymbol, 'SymbolLabelSpec'], None]

    def color_map(self) -> 'ColorMap':
        from omr.imageoperations.symbol_label_set import _color_map
        return _color_map(self.labels)


SYMBOL_DETECTION_HEAD_COUNT = 1


def symbol_detection_heads(label_sets: 'SymbolClassLabelSets') -> Tuple[SymbolHeadSpec, ...]:
    return (SymbolHeadSpec(
        name='note_types',
        labels=label_sets.note_type,
        draw_mask=draw_note_type_mask,
        apply_label=apply_note_type_label,
    ),)


def head_color_maps(label_sets: 'SymbolClassLabelSets') -> List['ColorMap']:
    return [h.color_map() for h in symbol_detection_heads(label_sets)]


def head_classes(label_sets: 'SymbolClassLabelSets') -> List[int]:
    return [len(h.labels) for h in symbol_detection_heads(label_sets)]
