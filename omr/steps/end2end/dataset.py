from typing import List, Optional

from database.file_formats.pcgts import PcGts
from omr.dataset import Dataset, DatasetParams, DatasetCallback
from omr.imageoperations import ImageOperationList

from .sample_builder import Sample, build_training_samples


class End2EndDataset(Dataset):
    @staticmethod
    def create_image_operation_list(params: DatasetParams) -> ImageOperationList:
        # Crops are built directly from the original page image by the sample builder,
        # since no image operation covers combined music+lyric regions.
        return ImageOperationList([])

    def __init__(self, pcgts: List[PcGts], params: DatasetParams):
        params.gt_required = True
        super().__init__(pcgts, params)

    def to_end2end_samples(self, callback: Optional[DatasetCallback] = None) -> List[Sample]:
        return build_training_samples(self.files, callback)
