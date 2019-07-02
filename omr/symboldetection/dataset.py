from database.file_formats.pcgts import PcGts, Line, PageScaleReference, BlockType
from omr.dataset import RegionLineMaskData
from typing import List, Generator, Tuple, Union, Optional
from database import DatabaseBook
import numpy as np
from typing import NamedTuple
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
    ImageOperationData, ImageRescaleToHeightOperation, ImagePadToPowerOf2, ImageDrawRegions, ImageApplyFCN

from omr.dewarping.dummy_dewarper import NoStaffLinesAvailable, NoStaffsAvailable

import logging
logger = logging.getLogger(__name__)


class Rect(NamedTuple):
    t: int
    b: int
    l: int
    r: int


class LoadedImage(NamedTuple):
    music_line: Line
    line_image: np.ndarray
    original_image: np.ndarray
    rect: Rect
    str_gt: List[str]


class ScaleImage(NamedTuple):
    img: np.ndarray
    order: int = 3


@dataclass
class SymbolDetectionDatasetParams:
    gt_required: bool = False
    height: int = 80
    dewarp: bool = False
    cut_region: bool = False
    pad: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int, int]] = (0, 80, 0, 80)
    pad_power_of_2: Optional[int] = 3
    center: bool = True
    staff_lines_only: bool = True
    masks_as_input: bool = False
    apply_fcn_pad_power_of_2: int = 3
    apply_fcn_model: Optional[str] = None
    apply_fcn_height: Optional[int] = None
    neume_types_only: bool = False


class SymbolDatasetCallback(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loading(self, n: int, total: int):
        pass

    @abstractmethod
    def loading_started(self, total: int):
        pass

    @abstractmethod
    def loading_finished(self, total: int):
        pass

    def apply(self, gen, total: int):
        self.loading_started(total)
        self.loading(0, total)
        for i, v in enumerate(gen):
            self.loading(i + 1, total)
            yield v
        self.loading_finished(total)


class SymbolDetectionDataset:
    def __init__(self, pcgts: List[PcGts], params: SymbolDetectionDatasetParams):
        self.files = pcgts
        self.params = params
        self.loaded: Optional[List[Tuple[Line, np.ndarray, str]]] = None
        self.marked_symbol_data: Optional[List[Tuple[Line, np.ndarray]]] = None

        operations = [
            ImageLoadFromPageOperation(invert=True, files=[('gray_norm_x2', False)]),
            ImageDrawRegions(block_types=[BlockType.DROP_CAPITAL] if params.cut_region else [], color=0),
            ImageExtractDewarpedStaffLineImages(params.dewarp, params.cut_region, params.pad, params.center, params.staff_lines_only),
        ]
        if self.params.apply_fcn_model is not None:
            operations += [
                ImageRescaleToHeightOperation(height=self.params.apply_fcn_height if self.params.apply_fcn_height else self.params.height),
                ImagePadToPowerOf2(self.params.apply_fcn_pad_power_of_2),
                ImageApplyFCN(self.params.apply_fcn_model),
            ]

        operations += [
            ImageRescaleToHeightOperation(height=params.height),
        ]

        if params.pad_power_of_2:
            operations.append(ImagePadToPowerOf2(params.pad_power_of_2))

        self.line_and_mask_operations = ImageOperationList(operations)

    def to_music_line_page_segmentation_dataset(self, callback: Optional[SymbolDatasetCallback] = None):
        from pagesegmentation.lib.dataset import Dataset, SingleData
        return Dataset([SingleData(image=d.line_image if self.params.cut_region else d.region, binary=None, mask=d.mask,
                                   user_data=d) for d in self.marked_symbols(callback)])

    def to_music_line_calamari_dataset(self, train=False, callback: Optional[SymbolDatasetCallback] = None):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        marked_symbols = self.marked_symbols(callback)

        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image.transpose()
            elif self.params.masks_as_input:
                hot = d.operation.images[3].image
                hot[:, :, 0] = (d.line_image if self.params.cut_region else d.region)
                return hot.transpose([1, 0, 2])
            else:
                return d.region.transpose()

        images = [get_input_image(d).astype(np.uint8) for d in marked_symbols]
        if self.params.neume_types_only:
            gts = [d.calamari_sequence().calamari_neume_types_str for d in marked_symbols]
        else:
            gts = [d.calamari_sequence().calamari_str for d in marked_symbols]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def marked_symbols(self, callback: Optional[SymbolDatasetCallback] = None) -> List[RegionLineMaskData]:
        if self.marked_symbol_data is None:
            self.marked_symbol_data = list(self._create_marked_symbols(callback))

        return self.marked_symbol_data

    def _create_marked_symbols(self, callback: Optional[SymbolDatasetCallback]) -> Generator[RegionLineMaskData, None, None]:
        def wrapper(g):
            if callback:
                return callback.apply(g, total=len(self.files))
            return g

        for f in tqdm(wrapper(self.files), total=len(self.files), desc="Loading music lines"):
            try:
                input = ImageOperationData([], PageScaleReference.NORMALIZED_X2, page=f.page)
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
    pages = [p for p in DatabaseBook('Graduel').pages()]
    params = SymbolDetectionDatasetParams(
        pad=(0, 10, 0, 40),
        dewarp=True,
        center=True,
        staff_lines_only=True,
        cut_region=False,
    )

    if not True:
        pages = pages[0:5]
        f, ax = plt.subplots(len(pages), 9, sharex='all', sharey='all')
        for i, p in enumerate(pages):
            pcgts = PcGts.from_file(p.file('pcgts'))
            dataset = SymbolDetectionDataset([pcgts], params)
            calamari_dataset = dataset.to_music_line_calamari_dataset()
            for a, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.marked_symbols())):
                img, region, mask = out.line_image, out.region, out.mask
                img = sample['image']
                ax[i, a].imshow(img)
    if True:
        page = pages[0]
        pcgts = PcGts.from_file(page.file('pcgts'))
        dataset = SymbolDetectionDataset([pcgts], params)
        ps_dataset = dataset.to_music_line_page_segmentation_dataset()
        f, ax = plt.subplots(len(ps_dataset.data), 3, sharex='all', sharey='all')
        for i, out in enumerate(ps_dataset.data):
            img, region, mask = out.image, out.image, out.mask
            # img = sample['image'][:,:,1:4].transpose([1,0,2])
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                print(mask.shape, mask.dtype, mask.min(), mask.max())
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(region)
                ax[i, 2].imshow(mask)
                # imsave("/home/wick/line0.jpg", 255 - (mask / mask.max() * 255))
    if not True:
        page = pages[0]
        pcgts = PcGts.from_file(page.file('pcgts'))
        dataset = SymbolDetectionDataset([pcgts], params)
        calamari_dataset = dataset.to_music_line_calamari_dataset()
        f, ax = plt.subplots(len(calamari_dataset.samples()), 3, sharex='all', sharey='all')
        for i, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.marked_symbols())):
            img, region, mask = out.line_image, out.region, out.mask
            # img = sample['image'][:,:,1:4].transpose([1,0,2])
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(sample['image'][:,:].transpose([1, 0]))
                ax[i, 2].imshow(mask)
                ax[i, 1].set_title(sample['text'])
                # imsave("/home/wick/line0.jpg", 255 - (mask / mask.max() * 255))

    plt.show()
