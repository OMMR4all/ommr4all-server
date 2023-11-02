from database.file_formats.pcgts import PcGts
import numpy as np
from typing import List
from omr.dataset import Dataset, DatasetParams, PageScaleReference
import logging

from omr.imageoperations import ImageLoadFromPageOperation, ImageOperationList, ImageScaleOperation, ImagePadToPowerOf2, \
    ImageExtractStaffLineImages, ImageOperationData

logger = logging.getLogger(__name__)


class PCDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        params.page_scale_reference = PageScaleReference.NORMALIZED
        return ImageOperationList([
            ImageLoadFromPageOperation(invert=True, files=[
                (params.page_scale_reference.file('gray' if params.gray else 'binary'), False)]),
            ImageExtractStaffLineImages(full_page=params.full_page, pad=params.pad,
                                        extract_region_only=params.extract_region_only,
                                        gt_line_thickness=params.gt_line_thickness),
            # ImageScaleOperation(0.5),  # Do not scale here, the line detector needs full resolution images
            ImagePadToPowerOf2(),  # Padding also done in line detector
        ])

    def __init__(self,
                 pcgts: List[PcGts],
                 params: DatasetParams,
                 ):
        super().__init__(pcgts, params)


class PCDatasetTorch(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        params.page_scale_reference = PageScaleReference.HIGHRES
        return ImageOperationList([
            ImageLoadFromPageOperation(invert=False, files=[(params.page_scale_reference.file('color'), False)]),
            ImageExtractStaffLineImages(full_page=params.full_page, pad=params.pad,
                                        extract_region_only=params.extract_region_only,
                                        gt_line_thickness=params.gt_line_thickness),
            # ImageScaleOperation(0.5),  # Do not scale here, the line detector needs full resolution images
            ImagePadToPowerOf2(),  # Padding also done in line detector
        ])

    def __init__(self,
                 pcgts: List[PcGts],
                 params: DatasetParams,
                 ):
        super().__init__(pcgts, params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from database import DatabaseBook

    page = DatabaseBook('Graduel').pages()[0]
    pcgts = PcGts.from_file(page.file('pcgts'))
    params = DatasetParams(
        full_page=True,
        gray=True,
        pad=(5, 5),
    )
    dataset = PCDataset([pcgts], params)
    images = dataset.load()

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
