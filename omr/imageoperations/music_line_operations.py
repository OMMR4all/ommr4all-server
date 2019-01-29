from .image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from .image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any, Optional
from omr.datatypes import Page, MusicLine, Neume, NoteComponent, Clef, ClefType, Accidental, AccidentalType, GraphicalConnectionType
import numpy as np
import PIL.ImageOps
from PIL import Image
from copy import copy
from enum import Enum
from omr.dewarping.dummy_dewarper import dewarp, transform
import logging

logger = logging.getLogger(__name__)


class SymbolLabel(Enum):
    BACKGROUND = 0
    NOTE_START = 1
    NOTE_LOOPED = 2
    NOTE_GAPPED = 3
    CLEF_C = 4
    CLEF_F = 5
    ACCID_NATURAL = 6
    ACCID_SHARP = 7
    ACCID_FLAT = 8


class ImageExtractDewarpedStaffLineImages(ImageOperation):
    def __init__(self):
        super().__init__()
        self.cropper = ImageCropToSmallestBoxOperation()

    def apply_single(self, data: ImageOperationData, debug=False) -> OperationOutput:
        image = data.images[0].image
        labels = np.zeros(image.shape, dtype=np.uint8)
        marked_symbols = np.zeros(labels.shape, dtype=np.uint8)
        i = 1
        s = []
        for mr in data.page.music_regions:
            for ml in mr.staffs:
                s.append(ml)
                ml.coords.draw(labels, i, 0, fill=True)
                self._symbols_to_mask(ml, marked_symbols)
                i += 1

        try:
            dew_page, dew_labels, dew_symbols = tuple(map(np.array, dewarp([Image.fromarray(image), Image.fromarray(labels), Image.fromarray(marked_symbols)], s, None)))
        except Exception as e:
            logger.exception("Exception during processing of page: {}".format(data.page.location.local_path()))
            raise e

        if debug:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(labels)
            ax[1].imshow(dew_labels)
            plt.show()

        i = 1
        out = []
        all_music_lines = []
        for mr in data.page.music_regions:
            all_music_lines += mr.staffs

        for mr in data.page.music_regions:
            for ml in mr.staffs:
                mask = dew_labels == i
                if np.sum(mask) != 0:  # empty mask, skip
                    img_data = copy(data)
                    img_data.page_image = image
                    img_data.music_region = mr
                    img_data.music_line = ml
                    img_data.images = [ImageData(mask, True), ImageData(dew_page, False), ImageData(dew_symbols, True)]
                    cropped = self.cropper.apply_single(img_data)[0]
                    self._extract_image_op(img_data)
                    r = self._resize_to_height(cropped.images, ml, rect=cropped.params)
                    if r is not None:  # Invalid resize (probably no staff lines present)
                        img_data.images, r_params = r
                        img_data.params = (i, cropped.params, r_params, all_music_lines)
                        out.append(img_data)

                i += 1

        return out

    def _resize_to_height(self, lines: List[ImageData], ml: MusicLine, rect, relative_staff_height=3) -> Optional[Tuple[List[ImageData], Any]]:
        if len(ml.staff_lines) < 1:
            return None

        t, b, l, r = rect
        for l in lines:
            assert(lines[0].image.shape == l.image.shape)

        height, width = lines[0].image.shape
        top = int(ml.staff_lines[0].center_y()) - t
        bot = int(ml.staff_lines[-1].center_y()) - t
        if top > height or bot > height:
            logger.error('Invalid line. Staff lines out of region.')
            return None
        staff_height = int(bot - top)
        pre_out_height = int(staff_height * relative_staff_height)
        pre_center = pre_out_height // 2
        pre_top = pre_center - staff_height // 2
        pre_bot = pre_top + staff_height
        top_to_add = pre_top - top
        bot_to_add = pre_out_height - top_to_add - height

        def single(t: ImageData) -> ImageData:
            out = copy(t)
            if top_to_add < 0:
                out.image = out.image[-top_to_add:, :]
            elif top_to_add > 0:
                out.image = np.vstack((np.zeros((top_to_add, width)), out.image))

            if bot_to_add < 0:
                out.image = out.image[:bot_to_add, :]
            elif bot_to_add > 0:
                out.image = np.vstack((out.image, np.zeros((bot_to_add, width))))

            if out.image.shape[0] != pre_out_height:
                raise Exception('Shape mismatch: {} != {}'.format(out.image.shape[0], pre_out_height))
            return out

        return [single(t) for t in lines], (top_to_add, )

    def _extract_image_op(self, data: ImageOperationData):
        data.images = [data.images[1], ImageData(data.images[0].image * data.images[1].image, False)] + data.images[2:]

    def _symbols_to_mask(self, ml: MusicLine, img: np.ndarray):
        if len(ml.staff_lines) < 2:  # at least two staff lines required
            return None

        radius = (ml.staff_lines[-1].center_y() - ml.staff_lines[0].center_y()) / len(ml.staff_lines) / 8

        def set(coord, label: SymbolLabel):
            img[int(coord.y - radius):int(coord.y + radius * 2), int(coord.x - radius): int(coord.x + radius * 2)] = label.value

        for s in ml.symbols:
            if isinstance(s, Neume):
                n: Neume = s
                for i, nc in enumerate(n.notes):
                    if i == 0:
                        set(nc.coord, SymbolLabel.NOTE_START)
                    elif nc.graphical_connection == GraphicalConnectionType.LOOPED:
                        set(nc.coord, SymbolLabel.NOTE_LOOPED)
                    else:
                        set(nc.coord, SymbolLabel.NOTE_GAPPED)

            elif isinstance(s, Clef):
                c: Clef = s
                if c.clef_type == ClefType.CLEF_F:
                    set(c.coord, SymbolLabel.CLEF_F)
                else:
                    set(c.coord, SymbolLabel.CLEF_C)
            elif isinstance(s, Accidental):
                a: Accidental = s
                if a.symbol_type == AccidentalType.NATURAL:
                    set(a.coord, SymbolLabel.ACCID_NATURAL)
                elif a.symbol_type == AccidentalType.FLAT:
                    set(a.coord, SymbolLabel.ACCID_FLAT)
                else:
                    set(a.coord, SymbolLabel.ACCID_SHARP)

        return img

    def local_to_global_pos(self, p: Point, params: Any):
        i, (t, b, l, r), (top, ), mls = params
        # default operations
        p = Point(p.x + l, t + p.y - top)
        # dewarp
        return Point(*transform(p.xy(), mls))


