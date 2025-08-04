from database.file_formats.pcgts import PcGts
from typing import List, Optional
from omr.dataset import DatasetParams
#from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
from omr.steps.symboldetection.sequencetosequence.params import CalamariParams
from database.model import Model
from dataclasses import dataclass
#from mashumaro import DataClassJSONMixin
from mashumaro.mixins.json import DataClassJSONMixin

from omr.steps.symboldetection.torchpixelclassifier.params import PageSegmentationTrainerTorchParams


@dataclass()
class AlgorithmTrainerParams(DataClassJSONMixin):
    n_epoch: int = 100
    n_iter: int = -1
    display: int = 100
    early_stopping_at_acc: float = 0
    early_stopping_test_interval: int = -1
    early_stopping_max_keep: int = -1
    l_rate: float = -1
    load: Optional[str] = None
    processes: int = -1
    train_data_multiplier: int = 1
    data_augmentation_factor: float = None

    def model_to_load(self) -> Optional[Model]:
        if not self.load:
            return None

        return Model.from_id_str(self.load)
    
    def mix_default(self, default_params: 'AlgorithmTrainerParams'):
        for key, value in default_params.to_dict().items():
            if type(getattr(self, key, None)) == str:
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(default_params, key))
            elif getattr(self, key, None) is None or getattr(self, key, -1) < 0:
                setattr(self, key, getattr(default_params, key))


@dataclass()
class AlgorithmTrainerSettings:
    dataset_params: DatasetParams
    train_data: List[PcGts]
    validation_data: List[PcGts]
    model: Optional[Model] = None
    params: Optional[AlgorithmTrainerParams] = None
    #page_segmentation_params: PageSegmentationTrainerParams = PageSegmentationTrainerParams()
    page_segmentation_torch_params: PageSegmentationTrainerTorchParams = PageSegmentationTrainerTorchParams()
    calamari_params: CalamariParams = CalamariParams()

