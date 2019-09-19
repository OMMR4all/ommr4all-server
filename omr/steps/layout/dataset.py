from database.file_formats.pcgts import PcGts, PageScaleReference
from typing import List

from omr.imageoperations import ImageOperationList

from omr.dataset import DatasetParams, Dataset

import logging
logger = logging.getLogger(__name__)


class LayoutDetectionDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        return ImageOperationList([])

    def __init__(self, pcgts: List[PcGts], params: DatasetParams):
        params.page_scale_reference = PageScaleReference.NORMALIZED
        super().__init__(pcgts, params)
