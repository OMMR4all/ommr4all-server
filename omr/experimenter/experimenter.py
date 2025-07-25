import os

from database.database_book_meta import DatabaseBookMeta
from database.file_formats import PcGts
from omr.dataset import DatasetParams
from omr.steps.algorithmtrainerparams import AlgorithmTrainerParams
from omr.steps.algorithmtypes import AlgorithmTypes
from copy import deepcopy

from omr.steps.symboldetection.torchpixelclassifier.params import PageSegmentationTrainerTorchParams

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
#from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
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
    ignore_gapped_symbols: bool = False


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
    output_debug_images: Optional[str]
    algorithm_type: AlgorithmTypes
    trainer_params: AlgorithmTrainerParams
    #page_segmentation_params: PageSegmentationTrainerParams
    page_segmentation_torch_params: PageSegmentationTrainerTorchParams
    calamari_params: CalamariParams
    calamari_dictionary_from_gt: bool


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


class ExperimenterScheduler:
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
        from omr.steps.step import Step
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
        #for i in train_args:
        #    logger.info(f"tf: {len(i.train_pcgts_files)} tv: {len(i.validation_pcgts_files)} tt: {len(i.test_pcgts_files)}")
        experimenter_class = Step.meta(self.global_args.algorithm_type).experimenter()
        results = [experimenter_class(args, logger).run_single() for args in train_args]
        experimenter_class.print_results(self.global_args, results, logger)


class Experimenter(ABC):
    def __init__(self, args: SingleDataArgs, parent_logger):
        self.args = args
        self.fold_log = parent_logger.getChild("fold_{}".format(args.id))

    def run_single(self):
        args = self.args
        fold_log = self.fold_log
        from omr.steps.algorithm import AlgorithmPredictorSettings, AlgorithmTrainerSettings, AlgorithmTrainerParams
        from omr.steps.step import Step
        global_args = args.global_args


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
            #fold_log.info(args.train_pcgts_files)
            #print(len(args.train_pcgts_files))
            #print(len(args.validation_pcgts_files))
            #print(len(args.test_pcgts_files))
            validation_data = args.validation_pcgts_files if args.validation_pcgts_files else args.train_pcgts_files
            fold_log.info(f"Starting training. Training:{len(args.train_pcgts_files)} and Val: {len(validation_data)}")

            trainer = Step.create_trainer(
                global_args.algorithm_type,
                AlgorithmTrainerSettings(
                    dataset_params=args.global_args.dataset_params,
                    train_data=args.train_pcgts_files,
                    validation_data=args.validation_pcgts_files if args.validation_pcgts_files else args.train_pcgts_files,
                    model=Model(MetaId.from_custom_path(model_path, global_args.algorithm_type)),
                    params=global_args.trainer_params,
                    #page_segmentation_params=global_args.page_segmentation_params,
                    page_segmentation_torch_params=global_args.page_segmentation_torch_params,
                    calamari_params=global_args.calamari_params,
                )
            )
            trainer.train()

        test_pcgts_files = args.test_pcgts_files
        if not global_args.skip_predict:
            fold_log.info("Starting prediction")
            pred_params = deepcopy(global_args.predictor_params)
            pred_params.modelId = MetaId.from_custom_path(model_path, global_args.algorithm_type)
            if global_args.calamari_dictionary_from_gt:
                words = set()
                for pcgts in test_pcgts_files:
                    words = words.union(sum([t.sentence.text().replace('-', '').split() for t in pcgts.page.all_text_lines()], []))
                pred_params.ctcDecoder.params.dictionary[:] = words

            pred = Step.create_predictor(
                global_args.algorithm_type,
                AlgorithmPredictorSettings(
                    None,
                    pred_params,
                ))
            full_predictions = list(pred.predict([f.page.location for f in test_pcgts_files]))
            predictions = self.extract_gt_prediction(full_predictions)
            with open(prediction_path, 'wb') as f:
                pickle.dump(predictions, f)
            if global_args.output_debug_images:

                self.output_debug_images(full_predictions)
                pass
            if global_args.output_book:
                fold_log.info("Outputting data")
                pred_book = DatabaseBook(global_args.output_book)
                if not pred_book.exists():
                    pred_book_meta = DatabaseBookMeta(pred_book.book, pred_book.book)
                    pred_book.create(pred_book_meta)

                output_pcgts = [PcGts.from_file(pcgts.page.location.copy_to(pred_book).file('pcgts'))
                                for pcgts in test_pcgts_files]

                self.output_prediction_to_book(pred_book, output_pcgts, full_predictions)

                for o_pcgts in output_pcgts:
                    o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())

        else:
            fold_log.info("Skipping prediction")

            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)

        predictions = tuple(predictions)
        if not global_args.skip_eval and len(predictions) > 0:
            fold_log.info("Starting evaluation")
            r = self.evaluate(predictions, global_args.evaluation_params)
        else:
            r = None

        # if not global_args.skip_cleanup:
        #    fold_log.info("Cleanup")
        #    shutil.rmtree(args.model_dir)

        return r

    @classmethod
    @abstractmethod
    def print_results(cls, args: GlobalDataArgs, results, log):
        pass

    @abstractmethod
    def extract_gt_prediction(self, full_predictions):
        pass

    @abstractmethod
    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
        pass
    @abstractmethod
    def output_debug_images(self, predictions):
        pass
    @abstractmethod
    def evaluate(self, predictions, evaluation_params):
        pass
