from database.file_formats.pcgts import PcGts
from typing import List, Optional
from omr.dataset import DatasetParams
from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
from omr.steps.symboldetection.sequencetosequence.params import CalamariParams
from database.model import Model
from dataclasses import dataclass
from mashumaro import DataClassJSONMixin


@dataclass()
class AlgorithmTrainerParams(DataClassJSONMixin):
    n_iter: int = -1
    display: int = 100
    early_stopping_test_interval: int = -1
    early_stopping_max_keep: int = -1
    l_rate: float = -1
    load: Optional[str] = None
    processes: int = -1

    def model_to_load(self) -> Optional[Model]:
        if not self.load:
            return None

        return Model.from_id(self.load)


@dataclass()
class AlgorithmTrainerSettings:
    dataset_params: DatasetParams
    train_data: List[PcGts]
    validation_data: List[PcGts]
    model: Optional[Model] = None
    params: Optional[AlgorithmTrainerParams] = None
    page_segmentation_params: PageSegmentationTrainerParams = PageSegmentationTrainerParams()
    calamari_params: CalamariParams = CalamariParams()

