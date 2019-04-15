import logging
from database import DatabaseBook, DatabaseBookMeta
from omr.dataset.datafiles import LockState, generate_dataset
from omr.symboldetection.dataset import SymbolDetectionDatasetParams, SymbolDetectionDataset, PcGts
from omr.symboldetection.evaluator import SymbolDetectionEvaluator, Counts, precision_recall_f1, AccCounts, SymbolDetectionEvaluatorParams
from omr.symboldetection.predictor import create_predictor, SymbolDetectionPredictorParameters, PredictorTypes
from pagesegmentation.lib.trainer import Trainer, TrainSettings
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
from omr.imageoperations.music_line_operations import SymbolLabel
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
    symbol_detection_params: SymbolDetectionDatasetParams
    symbol_evaluation_params: SymbolDetectionEvaluatorParams
    n_iter: int
    pretrained_model: Optional[str]
    data_augmentation: bool
    output_book: Optional[str]


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
    train_pcgts_dataset = SymbolDetectionDataset(args.train_pcgts_files, args.global_args.symbol_detection_params)
    validation_pcgts_dataset = SymbolDetectionDataset(args.validation_pcgts_files, args.global_args.symbol_detection_params) if args.validation_pcgts_files else None

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
        train_data = train_pcgts_dataset.to_music_line_page_segmentation_dataset()
        validation_data = validation_pcgts_dataset.to_music_line_page_segmentation_dataset() if validation_pcgts_dataset else train_data
        settings = TrainSettings(
            n_iter=global_args.n_iter,
            n_classes=len(SymbolLabel),
            l_rate=1e-4,
            train_data=train_data,
            validation_data=validation_data,
            display=100,
            load=args.global_args.pretrained_model,
            output=model_path,
            early_stopping_test_interval=500,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=8,
            data_augmentation=DefaultAugmenter(contrast=0.1, brightness=10, scale=(-0.1, 0.1, -0.1, 0.1)) if args.global_args.data_augmentation else None,
            checkpoint_iter_delta=None,
            compute_baseline=True,
        )

        trainer = Trainer(settings)
        trainer.train()

    if not global_args.skip_predict:
        fold_log.info("Starting prediction")
        pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER,
                                SymbolDetectionPredictorParameters([model_path], args.global_args.symbol_detection_params))
        full_predictions = list(pred.predict(args.test_pcgts_files))
        predictions = zip(*[(p.line.operation.music_line.symbols, p.symbols) for p in full_predictions])
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

            for p in full_predictions:
                o_pcgts = output_pcgts_by_page_name[p.line.operation.page.location.page]
                o_pcgts.page.music_line_by_id(p.line.operation.music_line.id).symbols = p.symbols

            for o_pcgts in output_pcgts:
                o_pcgts.to_file(o_pcgts.page.location.file('pcgts').local_path())
    else:
        fold_log.info("Skipping prediction")

        with open(prediction_path, 'rb') as f:
            predictions = pickle.load(f)

    if not global_args.skip_eval:
        fold_log.info("Starting evaluation")
        gt_symbols, pred_symbols = predictions
        evaluator = SymbolDetectionEvaluator(global_args.symbol_evaluation_params)
        metrics, counts, acc_counts, acc_acc, total_diffs = evaluator.evaluate(gt_symbols, pred_symbols)

        at = PrettyTable()

        at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])
        at.add_column("TP", counts[:, Counts.TP])
        at.add_column("FP", counts[:, Counts.FP])
        at.add_column("FN", counts[:, Counts.FN])

        prec_rec_f1 = np.array([precision_recall_f1(c[Counts.TP], c[Counts.FP], c[Counts.FN]) for c in counts])
        at.add_column("Precision", prec_rec_f1[:, 0])
        at.add_column("Recall", prec_rec_f1[:, 1])
        at.add_column("F1", prec_rec_f1[:, 2])
        fold_log.debug(at.get_string())

        at = PrettyTable()
        at.add_column("Type", ["Note all", "Note GC", "Note NS", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("True", acc_counts[:, AccCounts.TRUE])
        at.add_column("False", acc_counts[:, AccCounts.FALSE])
        at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
        at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

        fold_log.debug(at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "FP", "Acc", "Total"])
        at.add_row(total_diffs)
        fold_log.debug(at)

    else:
        prec_rec_f1 = None
        acc_acc = None
        total_diffs = None

    if not global_args.skip_cleanup:
        fold_log.info("Cleanup")
        shutil.rmtree(args.model_dir)

    return prec_rec_f1, acc_acc, total_diffs


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
            lock_states=[LockState('StaffLines', True), LockState('Layout', True)],
            n_train=n_train,
            val_amount=val_amount,
            cross_folds=cross_folds,
            single_folds=single_folds,
            train_books=train_books,
            test_books=test_books,
            train_books_extend=train_books_extend,
        )

        train_args = [SingleDataArgs(gd.fold,
                                     os.path.join(global_args.model_dir, 'symbol_detection_{}'.format(gd.fold)),
                                     gd.train_pcgts_files, gd.validation_pcgts_files,
                                     gd.test_pcgts_files,
                                     global_args) for gd in train_args]

        results = list(map(run_single, train_args))

        logger.info("Total Result:")

        prec_rec_f1_list = [r for r, _, _ in results]
        acc_counts_list = [r for _, r, _ in results]
        total_diffs = [r for _, _, r in results]

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        acc_mean = np.mean(acc_counts_list, axis=0)
        acc_std = np.std(acc_counts_list, axis=0)
        diffs_mean = np.mean(total_diffs, axis=0)
        diffs_std = np.std(total_diffs, axis=0)

        at = PrettyTable()

        at.add_column("Type", ["All", "All", "Notes", "Clefs", "Accids"])

        at.add_column("Precision", prf1_mean[:, 0])
        at.add_column("+-", prf1_std[:, 0])
        at.add_column("Recall", prf1_mean[:, 1])
        at.add_column("+-", prf1_std[:, 1])
        at.add_column("F1", prf1_mean[:, 2])
        at.add_column("+-", prf1_std[:, 2])
        logger.info("\n" + at.get_string())

        at = PrettyTable()
        at.add_column("Type", ["Note all", "Note GC", "Note NS", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("Accuracy [%]", acc_mean[:, 0] * 100)
        at.add_column("+- [%]", acc_std[:, 0] * 100)

        logger.info("\n" + at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "FP", "Acc", "Total"])
        at.add_row(diffs_mean)
        at.add_row(diffs_std)
        logger.info("\n" + at.get_string())

        if global_args.magic_prefix or True:
            # skip first all output
            all_symbol_detection = np.array(sum([[prf1_mean[1:, i], prf1_std[1:, i]] for i in range(3)], [])).transpose().reshape(-1)
            all_acc = np.array(np.transpose([acc_mean[:, 0], acc_std[:, 0]]).reshape([-1]))
            all_diffs = np.array(np.transpose([diffs_mean, diffs_std])).reshape([-1])
            print("{}{}".format(global_args.magic_prefix, ','.join(map(str, list(all_symbol_detection) + list(all_acc) + list(all_diffs)))))


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

    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--pad", type=int, default=[0], nargs="+")
    parser.add_argument("--center", action='store_true')
    parser.add_argument("--cut_region", action='store_true')
    parser.add_argument("--dewarp", action='store_true')
    parser.add_argument("--use_regions", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=1)

    # evaluation params
    parser.add_argument("--symbol_detected_min_distance", type=int, default=5)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.use_regions and args.cut_region:
        logger.warning("Cannot bot set 'cut_region' and 'staff_lines_only'. Setting 'cut_region=False'")
        args.cut_region = False

    global_args = GlobalDataArgs(
        args.magic_prefix,
        args.model_dir,
        args.cross_folds,
        args.single_folds,
        args.skip_train,
        args.skip_predict,
        args.skip_eval,
        not args.cleanup,
        SymbolDetectionDatasetParams(
            gt_required=True,
            height=args.height,
            pad=tuple(args.pad),
            center=args.center,
            cut_region=args.cut_region,
            dewarp=args.dewarp,
            staff_lines_only=not args.use_regions,
        ),
        symbol_evaluation_params=SymbolDetectionEvaluatorParams(
            symbol_detected_min_distance=args.symbol_detected_min_distance,
        ),
        n_iter=args.n_iter,
        pretrained_model=args.pretrained_model,
        data_augmentation=args.data_augmentation,
        output_book=args.output_book,
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
