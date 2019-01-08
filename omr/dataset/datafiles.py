from omr.datatypes.pcgts import PcGts
import random
import main.book as book
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class EmptyDataSetException(BaseException):
    pass


def dataset_by_locked_pages(n_train, lock_type, shuffle=True) -> Tuple[List[PcGts], List[PcGts]]:
    logger.info("Finding PcGts files with valid ground truth")
    pcgts = [PcGts.from_file(p.file('pcgts')) for p  in book.Book.list_all_pages_with_lock(lock_type)]

    if len(pcgts) == 0:
        raise EmptyDataSetException()

    if shuffle:
        random.shuffle(pcgts)

    train_pcgts = pcgts[:int(len(pcgts) * n_train)]
    val_pcgts = pcgts[len(train_pcgts):]

    if len(train_pcgts) == 0 or len(val_pcgts) == 0:
        raise EmptyDataSetException()

    return train_pcgts, val_pcgts


if __name__ == '__main__':
    print(dataset_by_locked_pages(0.5, 'Symbol'))
