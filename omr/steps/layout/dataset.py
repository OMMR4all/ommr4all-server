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
        return ImageOperationList([])

    def __init__(self, pcgts: List[PcGts], params: DatasetParams):
        params.page_scale_reference = PageScaleReference.NORMALIZED
        super().__init__(pcgts, params)
