import logging
from database import DatabaseBook
from database.database_book_meta import DatabaseBookMeta
from database.file_formats.performance.pageprogress import Locks
from database.model import MetaId, Model
from omr.dataset.datafiles import LockState, generate_dataset
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from omr.steps.symboldetection.dataset import DatasetParams, PcGts
from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, Counts, precision_recall_f1, AccCounts, SymbolDetectionEvaluatorParams
from omr.steps.step import Step, AlgorithmPredictor
from omr.steps.algorithm import AlgorithmPredictorSettings, AlgorithmTrainerSettings, AlgorithmTypes, AlgorithmTrainerParams
from omr.adapters.pagesegmentation.params import PageSegmentationTrainerParams
from typing import NamedTuple, List, Optional
import shutil
import os
import pickle
from prettytable import PrettyTable
import numpy as np

from omr.steps.symboldetection.sequencetosequence.params import CalamariParams

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
    symbol_detection_params: DatasetParams
    symbol_evaluation_params: SymbolDetectionEvaluatorParams
    n_iter: int
    pretrained_model: Optional[str]
    data_augmentation: bool
    output_book: Optional[str]
    symbol_detection_type: AlgorithmTypes
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
        from pagesegmentation.lib.data_augmenter import DefaultAugmenter
        fold_log.info("Starting training")
        trainer = Step.create_trainer(
            global_args.symbol_detection_type,
            AlgorithmTrainerSettings(
                dataset_params=args.global_args.symbol_detection_params,
                train_data=args.train_pcgts_files,
                validation_data=args.validation_pcgts_files if args.validation_pcgts_files else args.train_pcgts_files,
                model=Model(MetaId.from_custom_path(model_path, global_args.symbol_detection_type)),
                params=AlgorithmTrainerParams(
                    n_iter=global_args.n_iter,
                    display=100,
                    load=args.global_args.pretrained_model,
                    processes=8,
                ),
                page_segmentation_params=PageSegmentationTrainerParams(
                    data_augmenter=DefaultAugmenter(contrast=0.1, brightness=10, scale=(-0.1, 0.1, -0.1, 0.1)) if args.global_args.data_augmentation else None,
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

    #test_pcgts_files = [DatabaseBook('Graduel_Part_1').page('Graduel_de_leglise_de_Nevers_025').pcgts()]
    test_pcgts_files = args.test_pcgts_files
    if not global_args.skip_predict:
        fold_log.info("Starting prediction")
        pred = Step.create_predictor(
            global_args.symbol_detection_type,
            AlgorithmPredictorSettings(None, AlgorithmPredictorParams(MetaId.from_custom_path(model_path, global_args.symbol_detection_type))))
        full_predictions = list(pred.predict([f.page.location for f in test_pcgts_files]))
        predictions = zip(*[(p.line.operation.music_line.symbols, p.symbols) for p in sum([p.music_lines for p in full_predictions], [])])
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
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("True", acc_counts[:, AccCounts.TRUE])
        at.add_column("False", acc_counts[:, AccCounts.FALSE])
        at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
        at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

        fold_log.debug(at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
        at.add_row(total_diffs)
        fold_log.debug(at)

    else:
        prec_rec_f1 = None
        acc_acc = None
        total_diffs = None

    # if not global_args.skip_cleanup:
    #    fold_log.info("Cleanup")
    #    shutil.rmtree(args.model_dir)

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
                                     os.path.join(global_args.model_dir, 'symbol_detection_{}'.format(gd.fold)),
                                     gd.train_pcgts_files, gd.validation_pcgts_files,
                                     gd.test_pcgts_files,
                                     global_args) for gd in train_args]

        results = list(map(run_single, train_args))

        logger.info("Total Result:")

        prec_rec_f1_list = [r for r, _, _ in results if r is not None]
        acc_counts_list = [r for _, r, _ in results if r is not None]
        total_diffs = [r for _, _, r in results if r is not None]
        if len(prec_rec_f1_list) == 0:
            return

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
        at.add_column("Type", ["Note all", "Note PIS", "Clef type", "Clef PIS", "Accid type", "Sequence", "Sequence NC"])
        at.add_column("Accuracy [%]", acc_mean[:, 0] * 100)
        at.add_column("+- [%]", acc_std[:, 0] * 100)

        logger.info("\n" + at.get_string())

        at = PrettyTable(["Missing Notes", "Wrong NC", "Wrong PIS", "Missing Clefs", "Missing Accids", "Additional Notes", "FP Wrong NC", "FP Wrong PIS", "Additional Clefs", "Additional Accids", "Acc", "Total"])
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
    parser.add_argument("--type", type=lambda t: AlgorithmTypes[t],
                        choices=list(AlgorithmTypes),
                        default=AlgorithmTypes.SYMBOLS_PC)

    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--pad", type=int, default=[0], nargs="+")
    parser.add_argument("--pad_to_power_of_2", type=int, default=None)
    parser.add_argument("--center", action='store_true')
    parser.add_argument("--cut_region", action='store_true')
    parser.add_argument("--dewarp", action='store_true')
    parser.add_argument("--use_regions", action="store_true", default=False)
    parser.add_argument("--neume_types", action="store_true", default=False)

    parser.add_argument("--calamari_n_folds", type=int, default=0)
    parser.add_argument("--calamari_single_folds", type=int, nargs='+')
    parser.add_argument("--calamari_network", type=str, default='cnn=32:3x3,pool=2x2,cnn=64:3x3,pool=1x2,cnn=64:3x3,lstm=100,dropout=0.5')
    parser.add_argument("--calamari_channels", type=int, default=1)

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
        DatasetParams(
            gt_required=True,
            height=args.height,
            pad=list(args.pad),
            center=args.center,
            cut_region=args.cut_region,
            dewarp=args.dewarp,
            staff_lines_only=not args.use_regions,
            pad_power_of_2=args.pad_to_power_of_2,
            neume_types_only=args.neume_types,
        ),
        symbol_evaluation_params=SymbolDetectionEvaluatorParams(
            symbol_detected_min_distance=args.symbol_detected_min_distance,
        ),
        n_iter=args.n_iter,
        pretrained_model=args.pretrained_model,
        data_augmentation=args.data_augmentation,
        output_book=args.output_book,
        symbol_detection_type=args.type,
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
