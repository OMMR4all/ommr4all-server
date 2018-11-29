import main.book as book
from omr.datatypes import PcGts
from omr.datatypes.performance.pageprogress import PageProgress
from omr.symboldetection.pixelclassifier.trainer import PCTrainer
import logging
from random import shuffle

logger = logging.getLogger(__name__)


class EmptyDataSetException(BaseException):
    pass


class SymbolDetectionTrainer:
    def __init__(self, target_book: book.Book, n_train=0.8):
        logger.info("Finding PcGts files with valid ground truth")
        pcgts = []
        books = book.Book.list_available()
        for b in books:
            for p in b.pages():
                pp = PageProgress.from_json_file(p.file('page_progress', create_if_not_existing=True).local_path())
                if pp.locked.get('Symbol', False):
                    pcgts.append(PcGts.from_file(p.file('pcgts')))

        if len(pcgts) == 0:
            raise EmptyDataSetException()

        shuffle(pcgts)
        train_pcgts = pcgts[:int(len(pcgts) * n_train)]
        val_pcgts = pcgts[len(train_pcgts):]

        if len(train_pcgts) == 0 or len(val_pcgts) == 0:
            raise EmptyDataSetException

        logger.info("Starting training with {} training and {} validation files".format(len(train_pcgts), len(val_pcgts)))
        logger.debug("Training files: {}".format([p.page.location.local_path() for p in train_pcgts]))
        logger.debug("Validation files: {}".format([p.page.location.local_path() for p in val_pcgts]))
        trainer = PCTrainer(train_pcgts, val_pcgts)
        trainer.run(target_book)
        logger.info("Training finished for book {}".format(target_book.local_path()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    b = book.Book('demo')
    SymbolDetectionTrainer(b)
