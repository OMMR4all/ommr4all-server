from database.file_formats.pcgts import PcGts
import random
from database import DatabaseBook
from database.file_formats.performance import LockState
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class EmptyDataSetException(BaseException):
    pass


def dataset_by_locked_pages(n_train, locks: List[LockState], shuffle=True) -> Tuple[List[PcGts], List[PcGts]]:
    logger.info("Finding PcGts files with valid ground truth")
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in DatabaseBook.list_all_pages_with_lock(locks)]

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
    print(dataset_by_locked_pages(0.5, 'Symbol'))
