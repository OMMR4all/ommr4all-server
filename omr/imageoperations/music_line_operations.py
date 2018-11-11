from .image_operation import ImageOperation, ImageOperationData, OperationOutput, ImageData, Point, ImageOperationList
from .image_crop import ImageCropToSmallestBoxOperation
from typing import Tuple, List, NamedTuple, Any
from omr.datatypes import Page, MusicLine, Neume, NoteComponent, Clef, Accidental, AccidentalType, GraphicalConnectionType
import numpy as np
import PIL.ImageOps
from PIL import Image
from copy import copy

class ImageExtractDewarpedStaffLineImages(ImageOperationList):
    class SubOperation(ImageOperation):
        def __init__(self):
            super().__init__()

        def apply_single(self, data: ImageOperationData) -> OperationOutput:
            image = data.images[0].image
            labels = np.zeros(image.size[::-1], dtype=np.uint8)
            marked_symbols = np.zeros(labels.shape, dtype=np.uint8)
            i = 1
            s = []
            for mr in data.page.music_regions:
                for ml in mr.staffs:
                    s.append(ml)
                    ml.coords.draw(labels, i, 0, fill=True)
                    self._symbols_to_mask(ml, marked_symbols)
                    i += 1

            from omr.dewarping.dummy_dewarper import dewarp
            dew_page, dew_labels, dew_symbols = tuple(map(np.array, dewarp([image, Image.fromarray(labels), Image.fromarray(marked_symbols)], s, None)))

            i = 1
            out = []
            for mr in data.page.music_regions:
                for ml in mr.staffs:
                    mask = dew_labels == i

                    img_data = copy(data)
                    data.music_region = mr
                    data.music_line = ml
                    img_data.images = [ImageData(mask, True), ImageData(dew_page, False), ImageData(dew_symbols, True)]
                    img_data.params = (i, )
                    out.append(img_data)
                    i += 1

            return out

        def _symbols_to_mask(self, ml: MusicLine, img: np.ndarray):
            radius = (ml.staff_lines[-1].center_y() - ml.staff_lines[0].center_y()) / len(ml.staff_lines) / 4
            start_note_label = 1
            looped_label = 2
            gapped_label = 3
            clef_label = 4
            accid_label = 5

            def set(coord, label):
                img[int(coord.y - radius):int(coord.y + radius * 2), int(coord.x - radius): int(coord.x + radius * 2)] = label

            for s in ml.symbols:
                if isinstance(s, Neume):
                    n: Neume = s
                    for i, nc in enumerate(n.notes):
                        if i == 0:
                            set(nc.coord, start_note_label)
                        elif nc.graphical_connection == GraphicalConnectionType.LOOPED:
                            set(nc.coord, looped_label)
                        else:
                            set(nc.coord, gapped_label)

                elif isinstance(s, Clef):
                    c: Clef = s
                    set(c.coord, clef_label)
                elif isinstance(s, Accidental):
                    a: Accidental = s
                    set(a.coord, accid_label)

            return img

        def local_to_global_pos(self, p: Point, params: Any):
            pass

    class ExtractImageOp(ImageOperation):
        def __init__(self):
            super().__init__()

        def apply_single(self, data: ImageOperationData):
            data = copy(data)
            data.images = [data.images[1], ImageData(data.images[0].image * data.images[1].image, False)] + data.images[2:]
            data.params = None
            return [data]

    def __init__(self):
        super().__init__([
            ImageExtractDewarpedStaffLineImages.SubOperation(),
            ImageCropToSmallestBoxOperation(),
            ImageExtractDewarpedStaffLineImages.ExtractImageOp(),
        ])

