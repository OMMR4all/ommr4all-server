from typing import List
import numpy as np

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
    ImageOperationData, ImageRescaleToHeightOperation
from omr.imageoperations.textlineoperations import ImageExtractTextLineImages, ImageExtractDeskewedLyrics

from database.file_formats.pcgts import PcGts, PageScaleReference
from database.file_formats.pcgts.page.block import BlockType

from omr.dataset import DatasetParams, Dataset, ImageInput, LyricsNormalization

import logging
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        params.image_input = ImageInput.REGION_IMAGE
        params.page_scale_reference = PageScaleReference.NORMALIZED_X2

        operations = [
            ImageLoadFromPageOperation(invert=True, files=[(params.page_scale_reference.file(params.text_image_color_type), True)]),
            #ImageExtractTextLineImages({BlockType.LYRICS}, params.cut_region, params.pad),
            ImageExtractDeskewedLyrics(params.pad, params.text_types),
            ImageRescaleToHeightOperation(height=params.height),
        ]
        return ImageOperationList(operations)

    def __init__(self, pcgts: List[PcGts], params: DatasetParams):
        params.pad = (5, 10, 5, 20)
        params.dewarp = False
        params.staff_lines_only = True
        params.cut_region = True
        super().__init__(pcgts, params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from database.database_book import DatabaseBook
    from omr.dataset.dataset import LyricsNormalizationParams

    pages = [p for p in DatabaseBook('Gothic_Test').pages()]
    params = DatasetParams(
        height=60,
        gt_required=True,
        cut_region=True,
        pad=[0, 10, 0, 20],
        page_scale_reference=PageScaleReference.HIGHRES,
        lyrics_normalization=LyricsNormalizationParams(LyricsNormalization.ONE_STRING),
    )

    if True:
        page = pages[0]
        pcgts = PcGts.from_file(page.file('pcgts'))
        dataset = TextDataset([pcgts], params)
        calamari_dataset = dataset.to_text_line_calamari_dataset()
        f, ax = plt.subplots(len(calamari_dataset.samples()), 1, sharey='all')
        for i, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.load())):
            img, region, mask = out.line_image, out.region, out.mask
            print(img.shape, region.shape, mask)
            img = sample['image']
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                ax[i].imshow(img)
                ax[i].set_title(sample['text'])
                # imsave("/home/wick/line0.jpg", 255 - (mask / mask.max() * 255))

    plt.show()
