from database import DatabaseBook
from database.file_formats.pcgts import PcGts
import logging
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional
from omr.symboldetection.dataset import SymbolDetectionDatasetParams, SymbolDetectionDataset, SymbolDatasetCallback
from omr.symboldetection.pixelclassifier.params import SymbolDetectionPCParams
from omr.symboldetection.sequencetosequence.params import CalamariParams
from omr.symboldetection.predictor import PredictorTypes

logger = logging.getLogger(__name__)


class SymbolDetectionTrainerParams(NamedTuple):
    dataset_params: SymbolDetectionDatasetParams
    train_data: List[PcGts]
    validation_data: List[PcGts]
    n_iter: int = -1
    display: int = 100
    early_stopping_test_interval: int = -1
    early_stopping_max_keep: int = -1
    l_rate: float = -1
    load: Optional[str] = None
    load_base_dir: Optional[str] = None
    output: Optional[str] = None
    processes: int = -1
    page_segmentation_params: SymbolDetectionPCParams = SymbolDetectionPCParams()
    calamari_params: CalamariParams = CalamariParams()


class SymbolDetectionTrainerCallback(SymbolDatasetCallback):
    def __init__(self):
        super().__init__()
        self.total_iters = 0
        self.early_stopping_iters = 0

    def init(self, total_iters, early_stopping_iters):
        self.total_iters = total_iters
        self.early_stopping_iters = early_stopping_iters

    def next_iteration(self, iter: int, loss: float, acc: float):
        pass

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        pass

    def early_stopping(self):
        pass

    def resolving_files(self):
        pass


class SymbolDetectionTrainerBase(ABC):
    def __init__(self, params: SymbolDetectionTrainerParams):
        self.params = params
        self.train_dataset = SymbolDetectionDataset(params.train_data, params.dataset_params)
        self.validation_dataset = SymbolDetectionDataset(params.validation_data, params.dataset_params)

    @abstractmethod
    def run(self, model_for_book: Optional[DatabaseBook]=None, callback: Optional[SymbolDetectionTrainerCallback]=None):
        pass

    def preloaded_model_path(self) -> Optional[str]:
        if self.params.load_base_dir and not self.params.load:
            return self.__class__.load_base_dir_to_model(self.params.load_base_dir)
        else:
            return self.params.load

    @staticmethod
    @abstractmethod
    def load_base_dir_to_model(d: str) -> str:
        return d


def create_symbol_detection_trainer(
        type: PredictorTypes,
        params: SymbolDetectionTrainerParams):
    if type == PredictorTypes.PIXEL_CLASSIFIER:
        from omr.symboldetection.pixelclassifier.trainer import PCTrainer
        return PCTrainer(params)
    elif type == PredictorTypes.CALAMARI:
        from omr.symboldetection.sequencetosequence.trainer import OMRTrainer
        return OMRTrainer(params)
    elif type == PredictorTypes.PC_CALAMARI:
        from omr.symboldetection.pcs2s.trainer import PCS2STrainer
        return PCS2STrainer(params)
    else:
        raise ValueError("Unkown type for symbol detection trainer {}".format(type))
