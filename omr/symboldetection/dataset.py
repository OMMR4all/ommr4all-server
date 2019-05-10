from database.file_formats.pcgts import PcGts, MusicLine
from database.file_formats.pcgts.page.textregion import TextRegionType
from omr.dataset import RegionLineMaskData
from typing import List, Generator, Tuple, Union
from database import DatabaseBook
import numpy as np
from typing import NamedTuple
from tqdm import tqdm

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
    ImageOperationData, ImageRescaleToHeightOperation, ImagePadToPowerOf2, ImageDrawRegions

from omr.dewarping.dummy_dewarper import NoStaffLinesAvailable, NoStaffsAvailable

import logging
logger = logging.getLogger(__name__)


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


class ScaleImage(NamedTuple):
    img: np.ndarray
    order: int = 3


class SymbolDetectionDatasetParams(NamedTuple):
    gt_required: bool = False
    height: int = 80
    dewarp: bool = True
    cut_region: bool = True
    pad: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int, int]] = 0
    center: bool = True
    staff_lines_only: bool = False          # Determine the staff line AABB based on staff lines only (this is used for evaluation only)


class SymbolDetectionDataset:
    def __init__(self, pcgts: List[PcGts], params: SymbolDetectionDatasetParams):
        self.files = pcgts
        self.params = params
        self.loaded: List[Tuple[MusicLine, np.ndarray, str]] = None
        self.marked_symbol_data: List[Tuple[MusicLine, np.ndarray]] = None

        self.line_and_mask_operations = ImageOperationList([
            ImageLoadFromPageOperation(invert=True),
            ImageDrawRegions(text_region_types=[TextRegionType.DROP_CAPITAL] if params.cut_region else [], color=0),
            ImageExtractDewarpedStaffLineImages(params.dewarp, params.cut_region, params.pad, params.center, params.staff_lines_only),
            ImageRescaleToHeightOperation(height=params.height),
            ImagePadToPowerOf2(),
        ])

    def to_music_line_page_segmentation_dataset(self):
        from pagesegmentation.lib.dataset import Dataset, SingleData
        return Dataset([SingleData(image=d.line_image if self.params.cut_region else d.region, binary=None, mask=d.mask,
                                   user_data=d) for d in self.marked_symbols()])

    def to_music_line_calamari_dataset(self, train=False):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        marked_symbols = self.marked_symbols()
        images = [255 - (d.line_image if self.params.cut_region else d.region).astype(np.uint8) for d in marked_symbols]
        gts = [d.calamari_sequence().calamai_str for d in marked_symbols]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def marked_symbols(self) -> List[RegionLineMaskData]:
        if self.marked_symbol_data is None:
            self.marked_symbol_data = list(self._create_marked_symbols())

        return self.marked_symbol_data

    def _create_marked_symbols(self) -> Generator[RegionLineMaskData, None, None]:
        for f in tqdm(self.files, total=len(self.files), desc="Loading music lines"):
            try:
                input = ImageOperationData([], page=f.page)
                for outputs in self.line_and_mask_operations.apply_single(input):
                    yield RegionLineMaskData(outputs)
            except (NoStaffsAvailable, NoStaffLinesAvailable):
                pass
            except Exception as e:
                logger.exception("Exception during processing of page: {}".format(f.page.location.local_path()))
                raise e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    from imageio import imsave
    pages = [p for p in DatabaseBook('Graduel_Fully_Annotated').pages()]
    params = SymbolDetectionDatasetParams(
        height=80,
        gt_required=True,
        dewarp=True,
        cut_region=False,
        center=True,
        pad=(0, 10, 0, 20),
        staff_lines_only=True,
    )

    if True:
        pages = pages[0:5]
        f, ax = plt.subplots(len(pages), 9, sharex='all', sharey='all')
        for i, p in enumerate(pages):
            pcgts = PcGts.from_file(p.file('pcgts'))
            dataset = SymbolDetectionDataset([pcgts], params)
            calamari_dataset = dataset.to_music_line_calamari_dataset()
            for a, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.marked_symbols())):
                img, region, mask = out.line_image, out.region, out.mask
                img = sample['image']
                ax[i, a].imshow(img, cmap='gray')
    if False:
        page = pages[0]
        pcgts = PcGts.from_file(page.file('pcgts'))
        dataset = SymbolDetectionDataset([pcgts], params)
        calamari_dataset = dataset.to_music_line_calamari_dataset()
        f, ax = plt.subplots(len(calamari_dataset.samples()), 3, sharex='all')
        for i, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.marked_symbols())):
            img, region, mask = out.line_image, out.region, out.mask
            img = sample['image']
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                ax[i, 0].imshow(img, cmap='gray')
                ax[i, 1].imshow(sample['image'])
                ax[i, 2].imshow(img / 4 + mask * 50)
                ax[i, 1].set_title(sample['text'])
                # imsave("/home/wick/line0.jpg", 255 - (mask / mask.max() * 255))

    plt.show()
