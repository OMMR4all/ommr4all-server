from database import DatabaseBook
from database.file_formats.pcgts import PcGts
from omr.dataset.datafiles import dataset_by_locked_pages, LockState
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, NamedTuple, Optional
from omr.text.dataset import TextDatasetParams, TextDataset
from omr.symboldetection.sequencetosequence.params import CalamariParams
from omr.text.predictor import PredictorTypes

logger = logging.getLogger(__name__)


class TextTrainerParams(NamedTuple):
    dataset_params: TextDatasetParams
    train_data: List[PcGts]
    validation_data: List[PcGts]
    n_iter: int = -1
    display: int = 100
    early_stopping_test_interval: int = -1
    early_stopping_max_keep: int = -1
    l_rate: float = -1
    load: Optional[str] = None
    output: Optional[str] = None
    processes: int = -1
    calamari_params: CalamariParams = CalamariParams()


class TextTrainerCallback:
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


class TextTrainerBase(ABC):
    def __init__(self, params: TextTrainerParams):
        self.params = params
        self.train_dataset = TextDataset(params.train_data, params.dataset_params)
        self.validation_dataset = TextDataset(params.validation_data, params.dataset_params)

    @abstractmethod
    def run(self, model_for_book: Optional[DatabaseBook]=None, callback: Optional[TextTrainerCallback]=None):
        pass


def create_text_trainer(
        type: PredictorTypes,
        params: TextTrainerParams):
    if type == PredictorTypes.CALAMARI:
        from omr.text.calamari.trainer import CalamariTrainer
        return CalamariTrainer(params)
    else:
        raise ValueError("Unkown type for symbol detection trainer {}".format(type))


class TextTrainer:
    def __init__(self, target_book: DatabaseBook, n_train=0.8, callback: TextTrainerCallback = None):
        super().__init__()

        from omr.text.calamari.trainer import CalamariTrainer
        logger.info("Finding PcGts files with valid ground truth")
        train_pcgts, val_pcgts = dataset_by_locked_pages(n_train, [LockState('Symbols', True)])
        logger.info("Starting training with {} training and {} validation files".format(len(train_pcgts), len(val_pcgts)))
        logger.debug("Training files: {}".format([p.page.location.local_path() for p in train_pcgts]))
        logger.debug("Validation files: {}".format([p.page.location.local_path() for p in val_pcgts]))
        params = TextDatasetParams()
        trainer = create_symbol_detection_trainer(
            PredictorTypes.PIXEL_CLASSIFIER,
            TextTrainerParams(
                params,
                train_pcgts,
                val_pcgts,
            )
        )
        trainer.run(target_book, callback=callback)
        logger.info("Training finished for book {}".format(target_book.local_path()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    b = DatabaseBook('demo')
    TextTrainer(b)
