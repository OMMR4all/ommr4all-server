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


class LyricsLocationDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        params.image_input = ImageInput.REGION_IMAGE
        params.page_scale_reference = PageScaleReference.NORMALIZED_X2

        operations = [
            ImageLoadFromPageOperation(invert=False, files=[(params.page_scale_reference.file('color'), True)]),
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