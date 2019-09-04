from database.file_formats.pcgts import PcGts, PageScaleReference, BlockType
from typing import List
from database import DatabaseBook
import numpy as np

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
     ImageRescaleToHeightOperation, ImagePadToPowerOf2, ImageDrawRegions, ImageApplyFCN

from omr.dataset import DatasetParams, Dataset, ImageInput

import logging
logger = logging.getLogger(__name__)


class SymbolDetectionDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        params.image_input = ImageInput.REGION_IMAGE
        params.page_scale_reference = PageScaleReference.NORMALIZED_X2

        operations = [
            ImageLoadFromPageOperation(invert=True, files=[('gray_norm_x2', False)]),
            ImageDrawRegions(block_types=[BlockType.DROP_CAPITAL] if params.cut_region else [], color=0),
            ImageExtractDewarpedStaffLineImages(params.dewarp, params.cut_region, params.pad, params.center, params.staff_lines_only),
        ]
        if params.apply_fcn_model is not None:
            operations += [
                ImageRescaleToHeightOperation(height=params.apply_fcn_height if params.apply_fcn_height else params.height),
                ImagePadToPowerOf2(params.apply_fcn_pad_power_of_2),
                ImageApplyFCN(params.apply_fcn_model),
            ]

        operations += [
            ImageRescaleToHeightOperation(height=params.height),
        ]

        if params.pad_power_of_2:
            operations.append(ImagePadToPowerOf2(params.pad_power_of_2))

        return ImageOperationList(operations)

    def __init__(self, pcgts: List[PcGts], params: DatasetParams):
        params.pad = (0, 10, 0, 40)
        params.dewarp = False
        params.center = False
        params.staff_lines_only = True
        params.cut_region = False
        super().__init__(pcgts, params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    from imageio import imsave
    pages = [p for p in DatabaseBook('Graduel_Part_1').pages()]
    pages = [DatabaseBook('Graduel_Part_1').page('Graduel_de_leglise_de_Nevers_025')]
    params = DatasetParams(
        pad=[0, 10, 0, 40],
        dewarp=False,
        center=False,
        staff_lines_only=True,
        cut_region=False,
    )

    print(params.to_json())

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
        ps_dataset = dataset.to_page_segmentation_dataset()
        f, ax = plt.subplots(len(ps_dataset.data), 3, sharex='all', sharey='all')
        for i, out in enumerate(ps_dataset.data):
            img, region, mask = out.image, out.image, out.mask
            # img = sample['image'][:,:,1:4].transpose([1,0,2])
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                print(mask.shape, mask.dtype, mask.min(), mask.max())
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(np.minimum(region + mask * 10000, 255))
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
