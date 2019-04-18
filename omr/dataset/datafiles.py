from database.file_formats.pcgts import PcGts
import random
from database import DatabaseBook
from database.file_formats.performance import LockState
import logging
from typing import Tuple, List, Optional, NamedTuple

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


def flatten(data):
    out = []
    for d in data:
        out += d

    return out


def cross_fold(data, amount):
    folds = [data[i::amount] for i in range(amount)]
    return [(i, folds[i], flatten(folds[:i] + folds[i+1:])) for i in range(amount)]


class GeneratedData(NamedTuple):
    fold: int
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: Optional[List[PcGts]]
    test_pcgts_files: List[PcGts]


def generate_dataset(lock_states: List[LockState],
                     n_train: Optional[int],
                     val_amount: Optional[float],
                     cross_folds: Optional[int],
                     single_folds: Optional[List[int]],
                     train_books: Optional[List[str]],
                     test_books: Optional[List[str]],
                     train_books_extend: Optional[List[str]]) -> List[GeneratedData]:
    if n_train is None:
        n_train = -1

    if val_amount is None:
        val_amount = 0.2

    if cross_folds is None:
        cross_folds = 5

    if train_books_extend:
        train_pcgts_extend, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in train_books_extend])
        train_pcgts, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in train_books])

        if any([train_books, test_books]):
            if any([test_books]):
                logger.warning("You should only provide data to train books if you want to specify the books to use. Ignoring {}".format(test_books))

            all_pcgts, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in train_books])
        else:
            all_pcgts, _ = dataset_by_locked_pages(1, lock_states)

        logger.info("Starting experiment with {} files and {} extension files".format(len(all_pcgts), len(train_pcgts_extend)))

        def prepare_single_fold(fold, train_val_files, test_files, ext_files):
            if n_train >= 0:
                ext_files = ext_files[:n_train]

            if val_amount == 0:
                val, train = None, train_val_files
            else:
                _, val, train = cross_fold(train_val_files, int(1 / val_amount))[fold]
            return GeneratedData(fold, train + ext_files, val, test_files)

        fold_train_pcgts_extend = cross_fold(train_pcgts_extend, cross_folds)
        train_args = [
            prepare_single_fold(fold, all_pcgts, test_files, extend) for fold, test_files, extend in fold_train_pcgts_extend
        ]
    elif all([train_books, test_books]):
        train_pcgts, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in train_books])
        test_pcgts, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in test_books])
        if val_amount == 0:
            train_args = [
                GeneratedData(fold,
                              train if n_train < 0 else train[:n_train], None, test_pcgts,
                              )
                for fold, train, _ in cross_fold(train_pcgts, cross_folds)
            ]
        else:
            train_args = [
                GeneratedData(fold,
                              train, val, test_pcgts,
                              )
                for fold, val, train in cross_fold(train_pcgts, cross_folds)
            ]
    else:
        logger.info("Finding PcGts files with valid ground truth")
        if any([train_books, test_books]):
            if any([test_books]):
                logger.warning("You should only provide data to train books if you want to specify the books to use. Ignoring {}".format(test_books))

            all_pcgts, _ = dataset_by_locked_pages(1, lock_states, datasets=[DatabaseBook(book) for book in train_books])
        else:
            all_pcgts, _ = dataset_by_locked_pages(1, lock_states)

        logger.info("Starting experiment with {} files".format(len(all_pcgts)))

        def prepare_single_fold(fold, train_val_files, test_files):
            if n_train > 0:
                train_val_files = train_val_files[:n_train]

            if val_amount == 0:
                val, train = None, train_val_files
            else:
                _, val, train = cross_fold(train_val_files, int(1 / val_amount))[0]
            return GeneratedData(fold, train, val, test_files)

        train_args = [
            prepare_single_fold(fold, train_val_files, test_files) for fold, test_files, train_val_files in cross_fold(all_pcgts, cross_folds)
        ]

    train_args = [train_args[fold] for fold in (single_folds if single_folds and len(single_folds) > 0 else range(cross_folds))]

    return train_args


if __name__ == '__main__':
    import sys
    import random
    import numpy as np
    np.random.seed(1)
    random.seed(1)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)
    # print(dataset_by_locked_pages(0.5, [LockState('Symbols', True)], datasets=[DatabaseBook('demo')]))
    all_pages, _ = dataset_by_locked_pages(1, [], True, datasets=[DatabaseBook('Graduel_Part_3')])
    folds = [(i, [t.page.location.local_path() for t in test]) for i, test, train in cross_fold(all_pages, 5)]
    print([(i, pages) for i, pages in folds if any([('543' in p) for p in pages])])
