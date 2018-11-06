from omr.datatypes.pcgts import PcGts, MusicLine, Page, MusicRegion
from omr.datatypes.page.musicregion.musicline import Symbol, NoteComponent, Neume, Clef, Accidental, GraphicalConnectionType
from omr.symboldetection.thirdparty.calamari.calamari_ocr.ocr.data_processing.scale_to_height_processor import ScaleToHeightProcessor
from typing import List, Generator, Tuple
import main.book as book
import numpy as np
from PIL import Image
import PIL.ImageOps

class PcGtsDataset:
    def __init__(self, pcgts: List[PcGts], gt_required: bool):
        self.files = pcgts
        self.gt_required = gt_required
        self.loaded: List[Tuple[MusicLine, np.ndarray, str]] = None

    def music_lines(self) -> List[Tuple[MusicLine, np.ndarray, str]]:
        if self.loaded is None:
            self.loaded = list(self._load_music_lines())

        return self.loaded

    def _load_music_lines(self):
        for f in self.files:
            for l in self._load_images_of_file(f.page):
                yield l

    def _load_images_of_file(self, page: Page) -> Generator[Tuple[MusicLine, np.ndarray, str], None, None]:
        book_page = page.location
        bin = Image.open(book_page.file('binary_deskewed').local_path())
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

        def smallestbox(a, datas):
            r = a.any(1)
            m, n = a.shape
            c = a.any(0)
            q, w, e, r = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
            return [d[q:w, e:r] for d in datas], (q, w, e, r)

        i = 1
        for mr in page.music_regions:
            for ml in mr.staffs:
                mask = dew_labels == i

                out, rect = smallestbox(mask, [mask, dew_page])
                mask, line = tuple(out)
                # yield ml, line * np.stack([mask] * 3, axis=-1), str
                yield ml, self._resize_to_height(line * mask, ml, rect), self._symbols_to_string(ml.symbols)
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
                out.append(['b', 'n', 's'][a.accidental.value])

        return out

    def _resize_to_height(self, line: np.ndarray, ml: MusicLine, rect, relative_staff_height=3, absolute_pixel_size=80):
        t, b, l, r = rect
        height, width = line.shape
        top = int(ml.staff_lines[0].center_y()) - t
        bot = int(ml.staff_lines[-1].center_y()) - t
        staff_height = int(bot - top)
        pre_out_height = int(staff_height * relative_staff_height)
        pre_center = pre_out_height // 2
        pre_top = pre_center - staff_height // 2
        pre_bot = pre_top + staff_height
        top_to_add = pre_top - top
        bot_to_add = pre_out_height - top_to_add - height

        intermediate = np.vstack((np.zeros((top_to_add, width)), line, np.zeros((bot_to_add, width))))
        return ScaleToHeightProcessor.scale_to_h(intermediate, absolute_pixel_size) > 0.2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_023')
    pcgts = PcGts.from_file(page.file('pcgts'))
    dataset = PcGtsDataset([pcgts], True)
    images = dataset.music_lines()
    mls, raw_images, raw_gts = zip(*images)
    deskewed_images = [np.array(dewarp([Image.fromarray(img)], [ml])[0]) for ml, img in zip(mls, raw_images)]

    print(["".join(l) for l in raw_gts])

    f, ax = plt.subplots(len(images), 2, sharex='all')
    for i, (img, d) in enumerate(zip(raw_images, deskewed_images)):
        if np.min(img.shape) > 0:
            print(img.shape)
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(d)

    plt.show()
