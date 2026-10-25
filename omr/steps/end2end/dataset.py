from typing import List, Optional

from database.file_formats.pcgts import PcGts
from omr.dataset import Dataset, DatasetParams, DatasetCallback
from omr.imageoperations import ImageOperationList
from omr.imageoperations.symbol_label_set import SymbolClassLabelSets, SymbolLabelSpec

from .sample_builder import Sample, build_training_samples


def registered_class_specs(label_sets: Optional[SymbolClassLabelSets]) -> List[SymbolLabelSpec]:
    """The registered symbol classes of `label_sets`; each one gets its own codec tokens."""
    if label_sets is None:
        return []
    return [s for s in label_sets.main + label_sets.note_type if s.class_id is not None]


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
        class_ids = {s.class_id for s in registered_class_specs(self.params.symbol_label_sets)}
        return build_training_samples(self.files, self.params.height, class_ids, callback)
