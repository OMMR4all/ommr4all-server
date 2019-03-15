from database.file_formats.pcgts import PcGts, TextRegionType, MusicLine
import numpy as np
from typing import List, Tuple, Generator
from omr.dataset.pcgtsdataset import RegionLineMaskData
from tqdm import tqdm
import logging

from omr.imageoperations import ImageLoadFromPageOperation, ImageOperationList, ImageScaleOperation, ImagePadToPowerOf2, ImageExtractStaffLineImages, ImageOperationData


logger = logging.getLogger(__name__)


class PCDataset:
    def __init__(self, pcgts: List[PcGts], gt_required: bool,
                 full_page=True, gray=True,
                 extract_region_only=True, pad=0, gt_line_thickness=3,
                 ):
        self.gray = gray
        self.extract_region_only = extract_region_only
        self.files = pcgts
        self.gt_required = gt_required
        self.loaded: List[Tuple[MusicLine, np.ndarray, str]] = None
        self.marked_symbol_data: List[Tuple[MusicLine, np.ndarray]] = None

        self.line_and_mask_operations = ImageOperationList([
            ImageLoadFromPageOperation(invert=True, files=['gray_deskewed' if gray else 'binary_deskewed']),
            ImageExtractStaffLineImages(full_page=full_page, pad=pad, extract_region_only=extract_region_only, gt_line_thickness=gt_line_thickness),
            # ImageScaleOperation(0.5),  // Do not scale here, the line detector needs full resolution images
            # ImagePadToPowerOf2(),      // Padding also done in line detector
        ])

    def to_page_segmentation_dataset(self, target_staff_line_distance=10, origin_staff_line_distance=None):
        from pagesegmentation.lib.dataset import Dataset, SingleData, DatasetLoader
        loader = DatasetLoader(target_staff_line_distance)
        return Dataset([loader.load_images(
            SingleData(image=255 - d.line_image,
                       binary=((d.line_image < 125) * 255).astype(np.uint8),  # dummy binarization, it is not required, but to prevend NaN FgPA
                       mask=d.mask,
                       line_height_px=origin_staff_line_distance if origin_staff_line_distance is not None else d.operation.page.avg_staff_line_distance(),
                       original_shape=d.line_image.shape,
                       xpad=0,
                       ypad=0,
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
                input = ImageOperationData([], page=f.page)
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
    dataset = PCDataset([pcgts], True, full_page=False, gray=True, pad=5)
    images = dataset.marked_lines()

    f, ax = plt.subplots(len(images), 3, sharex='all')
    for i, out in enumerate(images):
        img, region, mask = out.line_image, out.region, out.mask
        if np.min(img.shape) > 0:
            print(img.shape)
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(region)
            ax[i, 2].imshow(img / 4 + mask * 50)

    plt.show()
