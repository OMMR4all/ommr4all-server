import os
import logging
from database import DatabaseBook
from database.database_book_meta import DatabaseBookMeta
from database.file_formats import PcGts
from database.file_formats.performance.pageprogress import Locks
from database.model import MetaId, Model
from omr.dataset import DatasetParams
from omr.dataset.datafiles import LockState, generate_dataset
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from omr.steps.symboldetection.sequencetosequence.params import CalamariParams
from omr.steps.text.evaluator import EvaluatorParams, Evaluator
from omr.steps.step import Step, AlgorithmPredictor
from omr.steps.algorithm import AlgorithmPredictorSettings, AlgorithmTrainerSettings, AlgorithmTypes, AlgorithmTrainerParams
from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
from typing import NamedTuple, List, Optional
import shutil
import os
import pickle
from prettytable import PrettyTable
import numpy as np

logger = logging.getLogger(__name__)


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
    n_iter: int
    pretrained_model: Optional[str]
    data_augmentation: bool
    output_book: Optional[str]
    detection_type: AlgorithmTypes
    calamari_network: str
    calamari_n_folds: int
    calamari_single_folds: Optional[List[int]]
    calamari_channels: int


class SingleDataArgs(NamedTuple):
    id: int
    model_dir: str
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: List[PcGts]
    test_pcgts_files: List[PcGts]

    global_args: GlobalDataArgs


def run_single(args: SingleDataArgs):
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
            global_args.detection_type,
            AlgorithmTrainerSettings(
                dataset_params=args.global_args.dataset_params,
                train_data=args.train_pcgts_files,
                validation_data=args.validation_pcgts_files if args.validation_pcgts_files else args.train_pcgts_files,
                model=Model(MetaId.from_custom_path(model_path, global_args.detection_type)),
                params=AlgorithmTrainerParams(
                    l_rate=1e-4,
                    n_iter=global_args.n_iter,
                    display=100,
                    load=args.global_args.pretrained_model,
                    processes=8,
                ),
                page_segmentation_params=PageSegmentationTrainerParams(
                    data_augmentation=args.global_args.data_augmentation,
                ),
                calamari_params=CalamariParams(
                    network=global_args.calamari_network,
                    n_folds=global_args.calamari_n_folds,
                    single_folds=global_args.calamari_single_folds,
                    channels=global_args.calamari_channels,
                ),
            )
        )
        trainer.train()

    test_pcgts_files = args.test_pcgts_files
    if not global_args.skip_predict:
        fold_log.info("Starting prediction")
        pred = Step.create_predictor(
            global_args.detection_type,
            AlgorithmPredictorSettings(None, AlgorithmPredictorParams(MetaId.from_custom_path(model_path, global_args.detection_type))))
        full_predictions = list(pred.predict([f.page.location for f in test_pcgts_files]))
        predictions = zip(*[(p.line.operation.text_line.sentence, p.text) for p in sum([p.text_lines for p in full_predictions], [])])
        with open(prediction_path, 'wb') as f:
            pickle.dump(predictions, f)

        if global_args.output_book:
            fold_log.info("Outputting data")

            pred_book = DatabaseBook(global_args.output_book)
            if not pred_book.exists():
                pred_book_meta = DatabaseBookMeta(global_args.output_book, global_args.output_book)
                pred_book.create(pred_book_meta)

            output_pcgts = [PcGts.from_file(pcgts.page.location.copy_to(pred_book).file('pcgts'))
                            for pcgts in args.test_pcgts_files]

            output_pcgts_by_page_name = {}
            for o_pcgts in output_pcgts:
                output_pcgts_by_page_name[o_pcgts.page.location.page] = o_pcgts
                for mr in o_pcgts.page.music_regions:
                    for ml in mr.staffs:
                        ml.symbols = []  # clear all symbols

                # clear all annotations
                o_pcgts.page.annotations.connections = []
                o_pcgts.page.comments.comments = []

            for p in full_predictions:
                o_pcgts = output_pcgts_by_page_name[p.line.operation.page.location.page]
                o_pcgts.page.music_line_by_id(p.line.operation.music_line.id).symbols = p.symbols

            for o_pcgts in output_pcgts:
                o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())
    else:
        fold_log.info("Skipping prediction")

        with open(prediction_path, 'rb') as f:
            predictions = pickle.load(f)

    predictions = tuple(predictions)
    if not global_args.skip_eval and len(predictions) > 0:
        fold_log.info("Starting evaluation")
        gt_sentence, pred_text_pos = predictions
        evaluator = Evaluator(global_args.evaluation_params)
        r = evaluator.evaluate(gt_sentence, pred_text_pos)
        at = PrettyTable(['#lines', '#chars', '#char errs', '#sync errs', 'CER'])
        at.add_row([r['total_instances'], r['total_chars'], r['total_char_errs'], r['total_sync_errs'], r['avg_ler']])
        print(at)
    else:
        r = {}

    # if not global_args.skip_cleanup:
        #    fold_log.info("Cleanup")
        #    shutil.rmtree(args.model_dir)

    return r


def flatten(data):
    out = []
    for d in data:
        out += d

    return out


def cross_fold(data, amount):
    folds = [data[i::amount] for i in range(amount)]
    return [(i, folds[i], flatten(folds[:i] + folds[i+1:])) for i in range(amount)]


class Experimenter:
    def __init__(self, args: GlobalDataArgs):
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
                                     os.path.join(global_args.model_dir, '{}_{}'.format(global_args.detection_type.value, gd.fold)),
                                     gd.train_pcgts_files, gd.validation_pcgts_files,
                                     gd.test_pcgts_files,
                                     global_args) for gd in train_args]

        results = list(map(run_single, train_args))

        logger.info("Total Result:")

        print(results)


if __name__ == "__main__":
    import sys
    import argparse
    import random
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument('--magic_prefix', default='EXPERIMENT_OUT=')
    parser.add_argument("--train", default=None, nargs="+")
    parser.add_argument("--test", default=None, nargs="+")
    parser.add_argument("--train_extend", default=None, nargs="+")
    parser.add_argument("--model_dir", type=str, default="model_out")
    parser.add_argument("--cross_folds", type=int, default=5)
    parser.add_argument("--single_folds", type=int, default=[0], nargs="+")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--cleanup", action="store_true", default=False)
    parser.add_argument("--n_train", default=-1, type=int)
    parser.add_argument("--n_iter", default=10000, type=int)
    parser.add_argument("--val_amount", default=0.2, type=float)
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--output_book", default=None, type=str)
    parser.add_argument("--type", type=lambda t: AlgorithmTypes[t],
                        choices=list(AlgorithmTypes),
                        default=AlgorithmTypes.OCR_CALAMARI)

    parser.add_argument("--calamari_n_folds", type=int, default=0)
    parser.add_argument("--calamari_single_folds", type=int, nargs='+')
    parser.add_argument("--calamari_network", type=str, default='cnn=32:3x3,pool=2x2,cnn=64:3x3,pool=1x2,cnn=64:3x3,lstm=100,dropout=0.5')
    parser.add_argument("--calamari_channels", type=int, default=1)

    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global_args = GlobalDataArgs(
        args.magic_prefix,
        args.model_dir,
        args.cross_folds,
        args.single_folds,
        args.skip_train,
        args.skip_predict,
        args.skip_eval,
        not args.cleanup,
        DatasetParams(
            gt_required=True,
        ),
        evaluation_params=EvaluatorParams(
        ),
        n_iter=args.n_iter,
        pretrained_model=args.pretrained_model,
        data_augmentation=args.data_augmentation,
        output_book=args.output_book,
        detection_type=args.type,
        calamari_n_folds=args.calamari_n_folds,
        calamari_single_folds=args.calamari_single_folds,
        calamari_network=args.calamari_network,
        calamari_channels=args.calamari_channels,
    )

    experimenter = Experimenter(global_args)
    experimenter.run(
        args.n_train,
        args.val_amount,
        args.cross_folds,
        args.single_folds,
        args.train,
        args.test,
        args.train_extend,
    )
