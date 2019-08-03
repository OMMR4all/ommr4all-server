from dataclasses import dataclass
from typing import Union, Tuple, List, Optional, Generator
import numpy as np
from omr.dataset import RegionLineMaskData

from omr.imageoperations import ImageExtractDewarpedStaffLineImages, ImageOperationList, ImageLoadFromPageOperation, \
    ImageOperationData, ImageRescaleToHeightOperation
from omr.imageoperations.textlineoperations import ImageExtractTextLineImages

from tqdm import tqdm

from database.file_formats.pcgts import PcGts, TextLine, PageScaleReference
from database.file_formats.pcgts.page.textregion import TextRegionType

import logging
logger = logging.getLogger(__name__)


@dataclass
class TextDatasetParams:
    gt_required: bool = False
    height: int = 80
    cut_region: bool = True
    pad: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int, int]] = 0
    scale_reference: PageScaleReference = PageScaleReference.NORMALIZED


class TextDataset:
    def __init__(self, pcgts: List[PcGts], params: TextDatasetParams):
        self.files = pcgts
        self.params = params
        self.loaded: Optional[List[Tuple[TextLine, np.ndarray, str]]] = None
        self._lines: Optional[List[Tuple[TextLine, np.ndarray]]] = None

        operations = [
            ImageLoadFromPageOperation(invert=True, files=[(params.scale_reference.file('binary'), True)]),
            ImageExtractTextLineImages({TextRegionType.LYRICS}, params.cut_region, params.pad),
            ImageRescaleToHeightOperation(height=params.height),
        ]

        self.line_and_mask_operations = ImageOperationList(operations)

    def to_text_line_calamari_dataset(self, train=False):
        from calamari_ocr.ocr.datasets.dataset import RawDataSet, DataSetMode
        lines = self.lines()

        def get_input_image(d: RegionLineMaskData):
            if self.params.cut_region:
                return d.line_image
            else:
                return d.region

        images = [255 - get_input_image(d).astype(np.uint8) for d in lines]
        gts = [self._extract_text(d) for d in lines]
        return RawDataSet(DataSetMode.TRAIN if train else DataSetMode.PREDICT, images=images, texts=gts)

    def lines(self) -> List[RegionLineMaskData]:
        if self._lines is None:
            self._lines = list(self._generator())

        return self._lines

    def _extract_text(self, data: RegionLineMaskData):
        text = data.operation.text_line.text(with_drop_capital=False)
        # text = text.replace('-', '')
        return text

    def _generator(self) -> Generator[RegionLineMaskData, None, None]:
        for f in tqdm(self.files, total=len(self.files), desc="Loading music lines"):
            try:
                input = ImageOperationData([], self.params.scale_reference, page=f.page, pcgts=f)
                for outputs in self.line_and_mask_operations.apply_single(input):
                    yield RegionLineMaskData(outputs)
            except Exception as e:
                logger.exception("Exception during processing of page: {}".format(f.page.location.local_path()))
                raise e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from database.database_book import DatabaseBook
    pages = [p for p in DatabaseBook('demo').pages()]
    params = TextDatasetParams(
        height=60,
        gt_required=True,
        cut_region=True,
        pad=(0, 10, 0, 20),
        scale_reference=PageScaleReference.HIGHRES,
    )

    if True:
        page = pages[0]
        pcgts = PcGts.from_file(page.file('pcgts'))
        dataset = TextDataset([pcgts], params)
        calamari_dataset = dataset.to_text_line_calamari_dataset()
        f, ax = plt.subplots(len(calamari_dataset.samples()), 1, sharey='all')
        for i, (sample, out) in enumerate(zip(calamari_dataset.samples(), dataset.lines())):
            img, region, mask = out.line_image, out.region, out.mask
            print(img.shape, region.shape, mask)
            # img = sample['image'][:,:,1:4].transpose([1,0,2])
            if np.min(img.shape) > 0:
                print(img.shape, img.dtype, img.min(), img.max())
                ax[i].imshow(img)
                ax[i].set_title(sample['text'])
                # imsave("/home/wick/line0.jpg", 255 - (mask / mask.max() * 255))

    plt.show()
