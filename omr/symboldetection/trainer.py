import main.book as book
from omr.datatypes import PcGts
from omr.datatypes.performance.pageprogress import PageProgress
from omr.dataset.datafiles import dataset_by_locked_pages
import logging
from random import shuffle

logger = logging.getLogger(__name__)


class SymbolDetectionTrainerCallback:
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



class SymbolDetectionTrainer:
    def __init__(self, target_book: book.Book, n_train=0.8, callback: SymbolDetectionTrainerCallback = None):
        super().__init__()

        from omr.symboldetection.pixelclassifier.trainer import PCTrainer
        logger.info("Finding PcGts files with valid ground truth")
        train_pcgts, val_pcgts = dataset_by_locked_pages(n_train, 'Symbol')
        logger.info("Starting training with {} training and {} validation files".format(len(train_pcgts), len(val_pcgts)))
        logger.debug("Training files: {}".format([p.page.location.local_path() for p in train_pcgts]))
        logger.debug("Validation files: {}".format([p.page.location.local_path() for p in val_pcgts]))
        trainer = PCTrainer(train_pcgts, val_pcgts)
        trainer.run(target_book, callback=callback)
        logger.info("Training finished for book {}".format(target_book.local_path()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    b = book.Book('demo')
    SymbolDetectionTrainer(b)
