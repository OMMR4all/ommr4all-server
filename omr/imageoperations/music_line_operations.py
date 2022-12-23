from omr.imageoperations.image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point
from omr.imageoperations.image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, Any, Optional
from database.file_formats.pcgts import Page, PageScaleReference, Line, MusicSymbol, ClefType, AccidType, \
    GraphicalConnectionType, Coords, SymbolType, BlockType
import numpy as np
from PIL import Image
from copy import copy
from enum import IntEnum
from omr.dewarping.dummy_dewarper import Dewarper, transform
import logging
import cv2

logger = logging.getLogger(__name__)


class SymbolLabel(IntEnum):
    BACKGROUND = 0
    NOTE_START = 1
    NOTE_LOOPED = 2
    NOTE_GAPPED = 3
    CLEF_C = 4
    CLEF_F = 5
    ACCID_NATURAL = 6
    ACCID_SHARP = 7
    ACCID_FLAT = 8

    def to_music_symbol(self) -> MusicSymbol:
        return {
            SymbolLabel.BACKGROUND: None,
            SymbolLabel.NOTE_START: MusicSymbol(SymbolType.NOTE,
                                                graphical_connection=GraphicalConnectionType.NEUME_START),
            SymbolLabel.NOTE_LOOPED: MusicSymbol(SymbolType.NOTE,
                                                 graphical_connection=GraphicalConnectionType.LOOPED),
            SymbolLabel.NOTE_GAPPED: MusicSymbol(SymbolType.NOTE,
                                                 graphical_connection=GraphicalConnectionType.GAPED),
            SymbolLabel.CLEF_C: MusicSymbol(SymbolType.CLEF, clef_type=ClefType.C),
            SymbolLabel.CLEF_F: MusicSymbol(SymbolType.CLEF, clef_type=ClefType.F),
            SymbolLabel.ACCID_NATURAL: MusicSymbol(SymbolType.ACCID, accid_type=AccidType.NATURAL),
            SymbolLabel.ACCID_SHARP: MusicSymbol(SymbolType.ACCID, accid_type=AccidType.SHARP),
            SymbolLabel.ACCID_FLAT: MusicSymbol(SymbolType.ACCID, accid_type=AccidType.FLAT),
        }[self]

    def get_color(self):
        return {0: [255, 255, 255],
                1: [255, 0, 0],
                2: [255, 120, 120],
                3: [120, 0, 0],
                4: [120, 255, 120],
                5: [0, 255, 0],
                6: [0, 0, 255],
                7: [50, 50, 255],
                8: [0, 0, 120]}[self.value]
# extract image of a staff line, and as mask, the highlighted staff lines
class ImageExtractStaffLineImages(ImageOperation):
    def __init__(self, full_page=True, pad=0, extract_region_only=True, gt_line_thickness=3):
        super().__init__()
        self.pad = pad
        self.gt_line_thickness = gt_line_thickness
        self.extract_region_only = extract_region_only
        self.cropper = ImageCropToSmallestBoxOperation(pad=pad)
        self.full_page = full_page

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        marked_regions = np.zeros(image.shape, dtype=np.uint8)
        marked_staff_lines = np.zeros(image.shape, dtype=np.uint8)
        page = data.page

        def scale(p):
            return page.page_to_image_scale(p, data.scale_reference)

        i = 1
        s = []
        for mr in data.page.music_blocks():
            for ml in mr.lines:
                s.append(ml)
                scale(ml.coords).draw(marked_regions, i, 0, fill=True)
                for sl in ml.staff_lines:
                    scale(sl.coords).draw(marked_staff_lines, color=1, thickness=self.gt_line_thickness)
                i += 1

        out = []
        if self.full_page:
            image_data = copy(data)
            image_data.images = [ImageData(marked_regions, True), ImageData(image, False), ImageData(marked_staff_lines, True)]
            image_data.params = None
            image_data.page_image = image
            image_data.music_lines = s
            out.append(image_data)
        else:
            i = 1

            for mr in data.page.music_regions:
                for ml in mr.staffs:
                    mask = marked_regions == i
                    if np.sum(mask) == 0:  # empty mask, skip
                        continue
                    else:
                        img_data = copy(data)
                        img_data.page_image = image
                        img_data.music_region = mr
                        img_data.music_line = ml
                        img_data.music_lines = [ml]
                        img_data.images = [ImageData(mask, True), ImageData(image, False), ImageData(marked_staff_lines, True)]
                        cropped = self.cropper.apply_single(img_data)[0]
                        self._extract_image_op(img_data)

                        img_data.params = (i, cropped.params)
                        out.append(img_data)

                    i += 1

        return out

    def _extract_image_op(self, data: ImageOperationData):
        data.images = [
                          data.images[1],
                          ImageData(data.images[0].image * data.images[1].image, False) if self.extract_region_only else data.images[1]
                      ] + data.images[2:]

    def local_to_global_pos(self, p: Point, params: Any):
        if self.full_page:
            return p
        else:
            i, (t, b, l, r) = params
            # default operations
            return Point(p.x + l, t + p.y)

# extract image of a staff line, and as mask, the highlighted staff lines
class ImageExtractDropCapitalsImages(ImageOperation):
    def __init__(self):
        super().__init__()

    def apply_single(self, data: ImageOperationData):
        image = data.images[0].image
        marked_drop_capitals = np.zeros(image.shape[:2], dtype=np.uint8)
        page = data.page

        def scale(p):
            return page.page_to_image_scale(p, data.scale_reference)

        i = 1
        s = []
        for mr in data.page.blocks_of_type(BlockType.DROP_CAPITAL):
            for drop_capital in mr.lines:
                s.append(drop_capital)
                scale(drop_capital.coords).draw(marked_drop_capitals, i, 0, fill=True)
                i += 1

        out = []
        image_data = copy(data)
        image_data.images = [ImageData(image, False), ImageData(marked_drop_capitals, True)]
        image_data.params = None
        image_data.page_image = image
        image_data.music_lines = s
        out.append(image_data)


        return out

    def _extract_image_op(self, data: ImageOperationData):
        data.images = [
                          data.images[0],
                          data.images[1]]

    def local_to_global_pos(self, p: Point, params: Any):
        return p


class ImageExtractDewarpedStaffLineImages(ImageOperation):
    def __init__(self, dewarp, cut_region, pad, center, staff_lines_only, keep_graphical_connection):
        super().__init__()
        self.dewarp = dewarp
        self.cut_region = cut_region
        self.center = center
        self.staff_lines_only = staff_lines_only
        self.cropper = ImageCropToSmallestBoxOperation(pad)
        self.keep_graphical_connection = keep_graphical_connection

    def apply_single(self, data: ImageOperationData, debug=False) -> OperationOutput:
        image = data.images[0].image
        labels = np.zeros(image.shape, dtype=np.uint8)
        marked_symbols = np.zeros(labels.shape, dtype=np.uint8)
        i = 1
        s: List[List[Coords]] = []

        def extract_transformed_coords(ml: Line) -> List[Coords]:
            lines = ml.staff_lines.sorted()
            return [data.page.page_to_image_scale(sl.coords, data.scale_reference) for sl in lines]

        for mr in data.page.music_blocks():
            for ml in mr.lines:
                coords = extract_transformed_coords(ml)
                s.append(coords)
                if self.staff_lines_only:
                    # draw staff lines instead of full area, however add an average line distance to top and bottom
                    avg_d = data.page.page_to_image_scale(ml.avg_line_distance(), data.scale_reference)
                    top = max(0, int(coords[0].points[:, 1].min() - avg_d))
                    bot = min(labels.shape[0], int(coords[-1].points[:, 1].max() + avg_d))
                    left = max(0, int(coords[0].points[:, 0].min()))
                    right = min(labels.shape[1], int(coords[0].points[:, 0].max()))
                    labels[top:bot, left:right] = i
                else:
                    data.page.page_to_image_scale(ml.coords, data.scale_reference).draw(labels, i, 0, fill=True)
                self._symbols_to_mask(ml, marked_symbols, data.page, data.scale_reference, self.keep_graphical_connection)
                i += 1

        if self.dewarp:
            images = [Image.fromarray(image), Image.fromarray(labels), Image.fromarray(marked_symbols)]
            dewarper = Dewarper(images[0].size, s)
            dew_page, dew_labels, dew_symbols = tuple(map(np.array, dewarper.dewarp(images, None)))
        else:
            dewarper = None
            dew_page, dew_labels, dew_symbols = image, labels, marked_symbols

        if debug:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(labels)
            ax[1].imshow(dew_labels)
            ax[2].imshow(dew_page)

            plt.show()

        i = 1
        out = []
        for mr in data.page.music_blocks():
            for ml in mr.lines:
                mask = dew_labels == i
                if np.sum(mask) != 0:  # empty mask, skip
                    img_data = copy(data)
                    img_data.page_image = image
                    img_data.music_region = mr
                    img_data.music_line = ml
                    img_data.images = [ImageData(mask, True), ImageData(dew_page, False), ImageData(dew_symbols, True)]
                    cropped = self.cropper.apply_single(img_data)[0]
                    self._extract_image_op(img_data)

                    if self.center:
                        coords = extract_transformed_coords(ml)
                        r = self._resize_to_height(cropped.images, coords, rect=cropped.params)
                        if r is not None:  # Invalid resize (probably no staff lines present)
                            img_data.images, r_params = r
                            img_data.params = (i, cropped.params, r_params, s, dewarper)
                            out.append(img_data)
                    else:
                        img_data.params = (i, cropped.params, (0, ), s, dewarper)
                        out.append(img_data)

                i += 1

        if debug:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(len(out), 3)
            for i, o in enumerate(out):
                ax[i, 0].imshow(o.images[0].image)
                ax[i, 1].imshow(o.images[1].image)
                ax[i, 2].imshow(o.images[2].image)

            plt.show()

        return out

    def _resize_to_height(self, lines: List[ImageData], coords: List[Coords], rect, relative_staff_height=3) -> Optional[Tuple[List[ImageData], Any]]:
        if len(coords) < 1:
            return None

        t, b, l, r = rect
        for l in lines:
            assert(lines[0].image.shape == l.image.shape)

        height, width = lines[0].image.shape
        top = int(coords[0].center_y()) - t
        bot = int(coords[-1].center_y()) - t
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
            fill_value = t.image.mean()
            out = copy(t)
            if top_to_add < 0:
                out.image = out.image[-top_to_add:, :]
            elif top_to_add > 0:
                out.image = np.vstack((np.full((top_to_add, width), fill_value, dtype=out.image.dtype), out.image))

            if bot_to_add < 0:
                out.image = out.image[:bot_to_add, :]
            elif bot_to_add > 0:
                out.image = np.vstack((out.image, np.full((bot_to_add, width), fill_value, dtype=out.image.dtype)))

            if out.image.shape[0] != pre_out_height:
                raise Exception('Shape mismatch: {} != {}'.format(out.image.shape[0], pre_out_height))
            return out

        return [single(t) for t in lines], (top_to_add, )

    def _extract_image_op(self, data: ImageOperationData):
        data.images = [data.images[1], ImageData(data.images[0].image * data.images[1].image, False)] + data.images[2:]

    def _symbols_to_mask(self, ml: Line, img: np.ndarray, page: Page, scale: PageScaleReference, keep_graphical_connection=None):
        if len(ml.staff_lines) < 2:  # at least two staff lines required
            return None

        def p2i(p):
            return page.page_to_image_scale(p, scale)

        radius = max(1, p2i(ml.staff_lines[-1].center_y() - ml.staff_lines[0].center_y()) / len(ml.staff_lines) / 8)

        def set(coord, label: SymbolLabel, dx=radius, dy=radius):
            coord = p2i(coord)
            # circle
            cv2.circle(img, tuple(coord.p.round().astype(int)), int(radius * 2), color=label.value, thickness=-1)
            # box
            # img[int(coord.y - dy):int(coord.y + dy * 2), int(coord.x - dx): int(coord.x + dx * 2)] = label.value

        for s in ml.symbols:
            if s.symbol_type == SymbolType.NOTE:
                if keep_graphical_connection and len(keep_graphical_connection) == 3:
                    if keep_graphical_connection[0] and s.graphical_connection == GraphicalConnectionType.NEUME_START:
                        set(s.coord, SymbolLabel.NOTE_START)
                    elif keep_graphical_connection[2] and s.graphical_connection == GraphicalConnectionType.LOOPED:
                        set(s.coord, SymbolLabel.NOTE_LOOPED)
                    else:
                        set(s.coord, SymbolLabel.NOTE_GAPPED)
                else:
                    if s.graphical_connection == GraphicalConnectionType.NEUME_START:
                        set(s.coord, SymbolLabel.NOTE_START)
                    elif s.graphical_connection == GraphicalConnectionType.LOOPED:
                        set(s.coord, SymbolLabel.NOTE_LOOPED)
                    else:
                        set(s.coord, SymbolLabel.NOTE_GAPPED)

            elif s.symbol_type == SymbolType.CLEF:
                if s.clef_type == ClefType.F:
                    set(s.coord, SymbolLabel.CLEF_F, dy=4 * radius)
                else:
                    set(s.coord, SymbolLabel.CLEF_C, dy=4 * radius)
            elif s.symbol_type == SymbolType.ACCID:
                if s.accid_type == AccidType.NATURAL:
                    set(s.coord, SymbolLabel.ACCID_NATURAL)
                elif s.accid_type == AccidType.FLAT:
                    set(s.coord, SymbolLabel.ACCID_FLAT)
                else:
                    set(s.coord, SymbolLabel.ACCID_SHARP)

        return img

    def local_to_global_pos(self, p: Point, params: Any):
        i, (t, b, l, r), (top, ), mls, dewarper = params
        # default operations
        p = Point(p.x + l, t + p.y - top)
        # dewarp
        if self.dewarp:
            return Point(*transform(p.xy(), mls))
        else:
            return p


if __name__ == "__main__":
    print(len(SymbolLabel))
