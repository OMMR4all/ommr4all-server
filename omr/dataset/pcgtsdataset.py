from omr.datatypes.pcgts import PcGts, MusicLine, Page, MusicRegion
from omr.datatypes.page.musicregion.musicline import Symbol, NoteComponent, Neume, Clef, Accidental, GraphicalConnectionType
from thirdparty.calamari.calamari_ocr.ocr.data_processing.scale_to_height_processor import ScaleToHeightProcessor
from typing import List, Generator, Tuple
import main.book as book
import numpy as np
from PIL import Image
import PIL.ImageOps
from typing import NamedTuple

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
    ImageOperationData, ImageRescaleToHeightOperation, ImagePadToPowerOf2

class Rect(NamedTuple):
    t: int
    b: int
    l: int
    r: int

class LoadedImage(NamedTuple):
    music_line: MusicLine
    line_image: np.ndarray
    original_image: np.ndarray
    rect: Rect
    str_gt: List[str]

class MusicLineAndMarkedSymbol:
    def __init__(self, op: ImageOperationData):
        self.operation = op
        self.line_image = op.images[1].image
        self.region = op.images[0].image
        self.mask = op.images[2].image


class ScaleImage(NamedTuple):
    img: np.ndarray
    order: int = 3


class PcGtsDataset:
    def __init__(self, pcgts: List[PcGts], gt_required: bool, height=80):
        self.files = pcgts
        self.gt_required = gt_required
        self.loaded: List[Tuple[MusicLine, np.ndarray, str]] = None
        self.height = height
        self.marked_symbol_data: List[Tuple[MusicLine, np.ndarray]] = None

        self.line_and_mask_operations = ImageOperationList([
            ImageLoadFromPageOperation(invert=True),
            ImageExtractDewarpedStaffLineImages(),
            ImageRescaleToHeightOperation(height=self.height),
            ImagePadToPowerOf2(),
        ])

    def to_music_line_page_segmentation_dataset(self):
        from thirdparty.page_segmentation.lib.dataset import Dataset, SingleData
        return Dataset([SingleData(image=d.line_image, binary=None, mask=d.mask,
                                   user_data=d) for d in self.marked_symbols()])

    def marked_symbols(self) -> List[MusicLineAndMarkedSymbol]:
        if self.marked_symbol_data is None:
            self.marked_symbol_data = list(self._create_marked_symbols())

        return self.marked_symbol_data

    def _create_marked_symbols(self) -> Generator[MusicLineAndMarkedSymbol, None, None]:
        for f in self.files:
            input = ImageOperationData([], page=f.page)
            for outputs in self.line_and_mask_operations.apply_single(input):

                # img, mask = self._resize_to_height([ScaleImage(loaded_image.images[0], 3), ScaleImage(mask, 0)], ml, loaded_image.rect)

                # mask = mask.astype(np.uint8)

                yield MusicLineAndMarkedSymbol(outputs)


    def music_lines(self) -> List[Tuple[MusicLine, np.ndarray, str]]:
        if self.loaded is None:
            self.loaded = list(self._load_music_lines())

        return self.loaded

    def to_calamari_dataset(self):
        from thirdparty.calamari.calamari_ocr.ocr.datasets import create_dataset, DataSetType, DataSetMode
        images = []
        gt = []
        for ml, img, s in self.music_lines():
            images.append(img)
            gt.append(s)

        dataset = create_dataset(
            DataSetType.RAW, DataSetMode.TRAIN,
            images, gt
        )
        dataset.load_samples()
        return dataset

    def _load_music_lines(self):
        for f in self.files:
            for ml, img, s, rect in self._load_images_of_file(f.page):
                yield ml, self._resize_to_height([ScaleImage(img), ], ml, rect)[0]

    def _load_images_of_file(self, page: Page) -> Generator[LoadedImage, None, None]:
        book_page = page.location
        bin = Image.open(book_page.file('gray_deskewed').local_path())
        bin = PIL.ImageOps.invert(bin)
        labels = np.zeros(bin.size[::-1], dtype=np.uint8)
        i = 1
        s = []
        for mr in page.music_regions:
            for ml in mr.staffs:
                s.append(ml)
                ml.coords.draw(labels, i, 0, fill=True)
                i += 1

        from omr.dewarping.dummy_dewarper import dewarp
        dew_page, dew_labels = tuple(map(np.array, dewarp([bin, Image.fromarray(labels)], s, None)))

        def smallestbox(a, datas) -> Tuple[List[np.ndarray], Rect]:
            r = a.any(1)
            m, n = a.shape
            c = a.any(0)
            q, w, e, r = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
            return [d[q:w, e:r] for d in datas], Rect(q, w, e, r)

        i = 1
        for mr in page.music_regions:
            for ml in mr.staffs:
                mask = dew_labels == i

                out, rect = smallestbox(mask, [mask, dew_page])
                mask, line = tuple(out)
                # yield ml, line * np.stack([mask] * 3, axis=-1), str
                yield LoadedImage(ml, line * mask, bin, rect, self._symbols_to_string(ml.symbols))
                i += 1

    def _symbols_to_string(self, symbols: List[Symbol]) -> List[str]:
        out = []
        for s in symbols:
            if isinstance(s, Neume):
                if out[-1] != ' ':
                    out.append(' ')

                n: Neume = s
                in_connection = False
                for i, nc in enumerate(n.notes):
                    out.append(str(nc.position_in_staff.value))
                    if i > 0:
                        if nc.graphical_connection == GraphicalConnectionType.LOOPED and not in_connection:
                            in_connection = True
                            out.insert(len(out) - 2, '(')
                        elif nc.graphical_connection == GraphicalConnectionType.GAPED and in_connection:
                            out.append(')')
                            in_connection = False

                if in_connection:
                    out.append(')')

                out.append(' ')
            elif isinstance(s, Clef):
                c: Clef = s
                out.append(['F', 'C'][c.clef_type.value] )
                out.append(str(c.position_in_staff.value))
            elif isinstance(s, Accidental):
                a: Accidental = s
                out.append(['n', 's', 'b'][a.accidental.value])  # 0, 1, -1

        return out

    def _resize_to_height(self, lines: List[ScaleImage], ml: MusicLine, rect, relative_staff_height=3) -> Tuple:
        t, b, l, r = rect
        for l in lines:
            assert(lines[0].img.shape == l.img.shape)

        height, width = lines[0].img.shape
        top = int(ml.staff_lines[0].center_y()) - t
        bot = int(ml.staff_lines[-1].center_y()) - t
        staff_height = int(bot - top)
        pre_out_height = int(staff_height * relative_staff_height)
        pre_center = pre_out_height // 2
        pre_top = pre_center - staff_height // 2
        pre_bot = pre_top + staff_height
        top_to_add = pre_top - top
        bot_to_add = pre_out_height - top_to_add - height

        def to_power_of_2(img: np.ndarray) -> np.ndarray:
            x, y = img.shape

            f = 2 ** 3
            tx = (((x // 2) // 2) // 2) * 8
            ty = (((y // 2) // 2) // 2) * 8

            if x % f != 0:
                px = tx - x + f
                x = x + px
            else:
                px = 0

            if y % f != 0:
                py = ty - y + f
                y = y + py
            else:
                py = 0

            pad = ((px, 0), (py, 0))
            return np.pad(img, pad, 'edge')

        def single(t: ScaleImage) -> np.ndarray:
            intermediate = np.vstack((np.zeros((top_to_add, width)), t.img, np.zeros((bot_to_add, width))))
            return ScaleToHeightProcessor.scale_to_h(intermediate, self.height, order=t.order)

        return tuple([to_power_of_2(single(t)) for t in lines])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    page = book.Book('demo').pages()[0]
    pcgts = PcGts.from_file(page.file('pcgts'))
    dataset = PcGtsDataset([pcgts], True)
    images = dataset.marked_symbols()

    f, ax = plt.subplots(len(images), 3, sharex='all')
    for i, out in enumerate(images):
        img, region, mask = out.line_image, out.region, out.mask
        if np.min(img.shape) > 0:
            print(img.shape)
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(region)
            ax[i, 2].imshow(img / 4 + mask * 50)

    plt.show()
