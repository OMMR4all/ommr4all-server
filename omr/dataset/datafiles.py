from database.file_formats.pcgts import PcGts
import random
from database import DatabaseBook
from database.file_formats.performance import LockState
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class EmptyDataSetException(BaseException):
    pass


def dataset_by_locked_pages(n_train, locks: List[LockState], shuffle=True, datasets: List[DatabaseBook] = None) -> Tuple[List[PcGts], List[PcGts]]:
    logger.info("Finding PcGts files with valid ground truth")
    pcgts = []
    for dataset in (datasets if datasets else DatabaseBook.list_available()):
        logger.debug("Listing files of dataset '{}'".format(dataset.book))
        if not dataset.exists():
            raise ValueError("Dataset '{}' does not exist at '{}'".format(dataset.book, dataset.local_path()))

        for page in dataset.pages_with_lock(locks):
            pcgts.append(PcGts.from_file(page.file('pcgts')))

    if len(pcgts) == 0:
        raise EmptyDataSetException()

    if shuffle:
        random.shuffle(pcgts)

    train_pcgts = pcgts[:int(len(pcgts) * n_train)]
    val_pcgts = pcgts[len(train_pcgts):]

    if 0 < n_train < 1 and (len(train_pcgts) == 0 or len(val_pcgts) == 0):
        raise EmptyDataSetException()

    return train_pcgts, val_pcgts


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)
    print(dataset_by_locked_pages(0.5, [LockState('Symbols', True)], datasets=[DatabaseBook('demo')]))
