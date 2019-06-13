from database.file_formats.pcgts import PcGts, TextRegionType, MusicLine, PageScaleReference
import numpy as np
from typing import List, Tuple, Generator, NamedTuple, Union
from omr.dataset import RegionLineMaskData
from tqdm import tqdm
import logging

from omr.imageoperations import ImageLoadFromPageOperation, ImageOperationList, ImageScaleOperation, ImagePadToPowerOf2, ImageExtractStaffLineImages, ImageOperationData


logger = logging.getLogger(__name__)


class StaffLineDetectionDatasetParams(NamedTuple):
    gt_required: bool = False
    full_page: bool = True
    gray: bool = True
    pad: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int, int]] = 0
    extract_region_only: bool = True
    gt_line_thickness: int = 2
    page_scale_reference: PageScaleReference = PageScaleReference.NORMALIZED


class PCDataset:
    def __init__(self, pcgts: List[PcGts],
                 params: StaffLineDetectionDatasetParams,
                 ):
        self.params = params
        self.files = pcgts
        self.loaded: List[Tuple[MusicLine, np.ndarray, str]] = None
        self.marked_symbol_data: List[Tuple[MusicLine, np.ndarray]] = None

        self.line_and_mask_operations = ImageOperationList([
            ImageLoadFromPageOperation(invert=True, files=[(params.page_scale_reference.file('gray' if params.gray else 'binary'), False)]),
            ImageExtractStaffLineImages(full_page=params.full_page, pad=params.pad, extract_region_only=params.extract_region_only, gt_line_thickness=params.gt_line_thickness),
            # ImageScaleOperation(0.5),  // Do not scale here, the line detector needs full resolution images
            # ImagePadToPowerOf2(),      // Padding also done in line detector
        ])

    def to_page_segmentation_dataset(self, target_staff_line_distance=10, origin_staff_line_distance=10):
        from pagesegmentation.lib.dataset import Dataset, SingleData, DatasetLoader
        loader = DatasetLoader(target_staff_line_distance)
        return Dataset([loader.load_images(
            SingleData(image=255 - d.line_image,
                       binary=((d.line_image < 125) * 255).astype(np.uint8),  # dummy binarization, it is not required, but to prevend NaN FgPA
                       mask=d.mask,
                       line_height_px=origin_staff_line_distance if origin_staff_line_distance is not None else d.operation.page.avg_staff_line_distance(),
                       original_shape=d.line_image.shape,
                       user_data=d)) for d in self.marked_lines()])

    def to_line_detection_dataset(self) -> List[RegionLineMaskData]:
        return self.marked_lines()

    def marked_lines(self) -> List[RegionLineMaskData]:
        if self.marked_symbol_data is None:
            self.marked_symbol_data = list(self._create_marked_lines())

        return self.marked_symbol_data

    def _create_marked_lines(self) -> Generator[RegionLineMaskData, None, None]:
        for f in tqdm(self.files, total=len(self.files), desc="Loading music lines"):
            try:
                input = ImageOperationData([], self.params.page_scale_reference, page=f.page)
                for outputs in self.line_and_mask_operations.apply_single(input):
                    yield RegionLineMaskData(outputs)
            except Exception as e:
                logger.exception("Exception during processing of page: {}".format(f.page.location.local_path()))
                raise e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from database import DatabaseBook
    page = DatabaseBook('Graduel').pages()[0]
    pcgts = PcGts.from_file(page.file('pcgts'))
    params = StaffLineDetectionDatasetParams(
        full_page=True,
        gray=True,
        pad=(5, 5),
    )
    dataset = PCDataset([pcgts], params)
    images = dataset.marked_lines()

    f, ax = plt.subplots(len(images), 3, sharex='all')
    for i, out in enumerate(images):
        img, region, mask = out.line_image, out.region, out.mask
        if np.min(img.shape) > 0:
            print(img.shape)
            if params.full_page:
                ax[0].imshow(img)
                ax[1].imshow(region)
                ax[2].imshow(mask)
            else:
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(region)
                ax[i, 2].imshow(img / 4 + mask * 50)

    plt.show()
