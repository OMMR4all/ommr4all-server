from omr.imageoperations.image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from omr.imageoperations.image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any, Optional
from database.file_formats.pcgts import Page, MusicLine, Neume, NoteComponent, ClefType, Clef, AccidentalType, Accidental, GraphicalConnectionType, MusicLines
import numpy as np
from PIL import Image
from copy import copy
from enum import IntEnum
from omr.dewarping.dummy_dewarper import dewarp, transform
import logging

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
        i = 1
        s = []
        for mr in data.page.music_regions:
            for ml in mr.staffs:
                s.append(ml)
                ml.coords.draw(marked_regions, i, 0, fill=True)
                ml.staff_lines.draw(marked_staff_lines, color=1, thickness=self.gt_line_thickness)
                i += 1

        out = []
        if self.full_page:
            image_data = copy(data)
            image_data.images = [ImageData(marked_regions, True), ImageData(image, False), ImageData(marked_staff_lines, True)]
            image_data.params = None
            image_data.page_image = image
            image_data.music_lines = MusicLines(s)
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
                        img_data.music_lines = MusicLines([ml])
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


class ImageExtractDewarpedStaffLineImages(ImageOperation):
    def __init__(self, dewarp, cut_region, pad, center, staff_lines_only):
        super().__init__()
        self.dewarp = dewarp
        self.cut_region = cut_region
        self.center = center
        self.staff_lines_only = staff_lines_only
        self.cropper = ImageCropToSmallestBoxOperation(pad)

    def apply_single(self, data: ImageOperationData, debug=False) -> OperationOutput:
        image = data.images[0].image
        labels = np.zeros(image.shape, dtype=np.uint8)
        marked_symbols = np.zeros(labels.shape, dtype=np.uint8)
        i = 1
        s = []
        for mr in data.page.music_regions:
            for ml in mr.staffs:
                s.append(ml)
                if self.staff_lines_only:
                    # draw staff lines instead of full area, however add an average line distance to top and bottom
                    lines = ml.staff_lines.sorted()
                    avg_d = ml.avg_line_distance()
                    top = int(lines[0].coords.points[:,1].min() - avg_d)
                    bot = int(lines[-1].coords.points[:, 1].max() + avg_d)
                    left = int(lines[0].coords.points[:, 0].min())
                    right = int(lines[0].coords.points[:, 0].max())
                    labels[top:bot, left:right] = i
                else:
                    ml.coords.draw(labels, i, 0, fill=True)
                self._symbols_to_mask(ml, marked_symbols)
                i += 1

        if self.dewarp:
            dew_page, dew_labels, dew_symbols = tuple(map(np.array, dewarp([Image.fromarray(image), Image.fromarray(labels), Image.fromarray(marked_symbols)], s, None)))
        else:
            dew_page, dew_labels, dew_symbols = image, labels, marked_symbols

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
                    if self.center:
                        r = self._resize_to_height(cropped.images, ml, rect=cropped.params)
                        if r is not None:  # Invalid resize (probably no staff lines present)
                            img_data.images, r_params = r
                            img_data.params = (i, cropped.params, r_params, all_music_lines)
                            out.append(img_data)
                    else:
                        img_data.params = (i, cropped.params, (0, ), all_music_lines)
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
                if a.accidental == AccidentalType.NATURAL:
                    set(a.coord, SymbolLabel.ACCID_NATURAL)
                elif a.accidental == AccidentalType.FLAT:
                    set(a.coord, SymbolLabel.ACCID_FLAT)
                else:
                    set(a.coord, SymbolLabel.ACCID_SHARP)

        return img

    def local_to_global_pos(self, p: Point, params: Any):
        i, (t, b, l, r), (top, ), mls = params
        # default operations
        p = Point(p.x + l, t + p.y - top)
        # dewarp
        if self.dewarp:
            return Point(*transform(p.xy(), mls))
        else:
            return p


if __name__ == "__main__":
    print(len(SymbolLabel))
