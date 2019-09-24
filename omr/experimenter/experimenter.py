import os

from database.database_book_meta import DatabaseBookMeta
from database.file_formats import PcGts
from omr.dataset import DatasetParams
from omr.steps.algorithmtrainerparams import AlgorithmTrainerParams
from omr.steps.algorithmtypes import AlgorithmTypes
from copy import deepcopy

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

import logging
from database import DatabaseBook
from database.file_formats.performance.pageprogress import Locks
from database.model import MetaId, Model
from omr.dataset.datafiles import LockState, generate_dataset
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
from typing import NamedTuple, List, Optional
import os
import pickle
from abc import ABC, abstractmethod

from omr.steps.symboldetection.sequencetosequence.params import CalamariParams

logger = logging.getLogger(__name__)


class EvaluatorParams(NamedTuple):
    debug: bool = False

    symbol_detected_min_distance: int = 0.005

    line_hit_overlap_threshold: float = 0.5
    staff_n_lines_threshold: int = 2
    staff_line_found_distance: int = 3


class GlobalDataArgs(NamedTuple):
    magic_prefix: Optional[str]
    model_dir: str
    cross_folds: int
    single_folds: Optional[List[int]]
    skip_train: bool
    skip_predict: bool
    skip_eval: bool
    skip_cleanup: bool
    dataset_params: DatasetParams
    evaluation_params: EvaluatorParams
    predictor_params: AlgorithmPredictorParams
    output_book: Optional[str]
    algorithm_type: AlgorithmTypes
    trainer_params: AlgorithmTrainerParams
    page_segmentation_params: PageSegmentationTrainerParams
    calamari_params: CalamariParams


class SingleDataArgs(NamedTuple):
    id: int
    model_dir: str
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: List[PcGts]
    test_pcgts_files: List[PcGts]

    global_args: GlobalDataArgs


def flatten(data):
    out = []
    for d in data:
        out += d

    return out


def cross_fold(data, amount):
    folds = [data[i::amount] for i in range(amount)]
    return [(i, folds[i], flatten(folds[:i] + folds[i+1:])) for i in range(amount)]


class Experimenter(ABC):
    def __init__(self, args: GlobalDataArgs):
        super().__init__()
        self.global_args = args

    def run(self,
            n_train: int,
            val_amount: float,
            cross_folds: int,
            single_folds: List[int],
            train_books: List[str],
            test_books: List[str],
            train_books_extend: List[str]
            ):
        global_args = self.global_args
        logger.info("Finding PcGts files with valid ground truth")
        train_args = generate_dataset(
            lock_states=[LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True)],
            n_train=n_train,
            val_amount=val_amount,
            cross_folds=cross_folds,
            single_folds=single_folds,
            train_books=train_books,
            test_books=test_books,
            train_books_extend=train_books_extend,
        )

        train_args = [SingleDataArgs(gd.fold,
                                     os.path.join(global_args.model_dir, '{}_{}'.format(global_args.algorithm_type.value, gd.fold)),
                                     gd.train_pcgts_files, gd.validation_pcgts_files,
                                     gd.test_pcgts_files,
                                     global_args) for gd in train_args]

        results = list(map(self.__class__.run_single, train_args))
        self.print_results(results, logger)

    @classmethod
    def run_single(cls, args: SingleDataArgs):
        from omr.steps.algorithm import AlgorithmPredictorSettings, AlgorithmTrainerSettings, AlgorithmTrainerParams
        from omr.steps.step import Step
        global_args = args.global_args

        fold_log = logger.getChild("fold_{}".format(args.id))

        def print_dataset_content(files: List[PcGts], label: str):
            fold_log.debug("Got {} {} files: {}".format(len(files), label, [f.page.location.local_path() for f in files]))

        print_dataset_content(args.train_pcgts_files, 'training')
        if args.validation_pcgts_files:
            print_dataset_content(args.validation_pcgts_files, 'validation')
        else:
            fold_log.debug("No validation data. Using training data instead")
        print_dataset_content(args.test_pcgts_files, 'testing')

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        prediction_path = os.path.join(args.model_dir, 'pred.json')
        model_path = os.path.join(args.model_dir, 'best')

        if not global_args.skip_train:
            fold_log.info("Starting training")
            trainer = Step.create_trainer(
                global_args.algorithm_type,
                AlgorithmTrainerSettings(
                    dataset_params=args.global_args.dataset_params,
                    train_data=args.train_pcgts_files,
                    validation_data=args.validation_pcgts_files if args.validation_pcgts_files else args.train_pcgts_files,
                    model=Model(MetaId.from_custom_path(model_path, global_args.algorithm_type)),
                    params=global_args.trainer_params,
                    page_segmentation_params=global_args.page_segmentation_params,
                    calamari_params=global_args.calamari_params,
                )
            )
            trainer.train()

        test_pcgts_files = args.test_pcgts_files
        if not global_args.skip_predict:
            fold_log.info("Starting prediction")
            pred_params = deepcopy(global_args.predictor_params)
            pred_params.modelId = MetaId.from_custom_path(model_path, global_args.algorithm_type)
            pred = Step.create_predictor(
                global_args.algorithm_type,
                AlgorithmPredictorSettings(
                    None,
                    pred_params,
                ))
            full_predictions = list(pred.predict([f.page.location for f in test_pcgts_files]))
            predictions = cls.extract_gt_prediction(full_predictions)
            with open(prediction_path, 'wb') as f:
                pickle.dump(predictions, f)

            if global_args.output_book:
                fold_log.info("Outputting data")
                pred_book = DatabaseBook(global_args.output_book)
                if not pred_book.exists():
                    pred_book_meta = DatabaseBookMeta(pred_book.book, pred_book.book)
                    pred_book.create(pred_book_meta)

                output_pcgts = [PcGts.from_file(pcgts.page.location.copy_to(pred_book).file('pcgts'))
                                for pcgts in test_pcgts_files]

                cls.output_prediction_to_book(pred_book, output_pcgts, full_predictions)

                for o_pcgts in output_pcgts:
                    o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())

        else:
            fold_log.info("Skipping prediction")

            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)

        predictions = tuple(predictions)
        if not global_args.skip_eval and len(predictions) > 0:
            fold_log.info("Starting evaluation")
            r = cls.evaluate(predictions, global_args.evaluation_params, fold_log)
        else:
            r = None

        # if not global_args.skip_cleanup:
        #    fold_log.info("Cleanup")
        #    shutil.rmtree(args.model_dir)

        return r

    @abstractmethod
    def print_results(self, results, log):
        pass

    @classmethod
    @abstractmethod
    def extract_gt_prediction(cls, full_predictions):
        pass

    @classmethod
    @abstractmethod
    def output_prediction_to_book(cls, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, predictions, evaluation_params, log):
        pass
