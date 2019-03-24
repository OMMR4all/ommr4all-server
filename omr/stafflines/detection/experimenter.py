import logging
from database.file_formats import PcGts
from database.file_formats.performance import LockState
from pagesegmentation.lib.trainer import TrainSettings, Trainer
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
from omr.dataset.datafiles import dataset_by_locked_pages
from omr.stafflines.detection.dataset import PCDataset, StaffLineDetectionDatasetParams
from omr.stafflines.detection.predictor import create_staff_line_predictor, StaffLinesModelType, StaffLinePredictorParameters
from omr.stafflines.detection.evaluator import StaffLineDetectionEvaluator, EvaluationData
from typing import NamedTuple, List, Optional
import os
import shutil
from prettytable import PrettyTable
import numpy as np
import json

logger = logging.getLogger(__name__)


class GlobalArgs(NamedTuple):
    model_dir: str
    cross_folds: int
    single_folds: Optional[List[int]]
    n_train: int

    target_line_space: int
    origin_line_space: Optional[int]
    do_not_use_model: bool
    post_processing: bool
    data_augmentation: bool
    smooth_staff_lines: int
    line_fit_distance: float

    skip_train: bool
    skip_prediction: bool
    skip_eval: bool
    skip_cleanup: bool

    dataset_params: StaffLineDetectionDatasetParams


class SingleDataArgs(NamedTuple):
    id: int
    output: str
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: List[PcGts]
    test_pcgts_files: List[PcGts]

    global_args: GlobalArgs


def run_single(args: SingleDataArgs):
    fold_log = logger.getChild("fold_{}".format(args.id))
    train_pcgts_dataset = PCDataset(
        args.train_pcgts_files if args.global_args.n_train < 0 else args.train_pcgts_files[:args.global_args.n_train],
        args.global_args.dataset_params)
    validation_pcgts_dataset = PCDataset(args.validation_pcgts_files, args.global_args.dataset_params)

    def print_dataset_content(files: List[PcGts], label: str):
        fold_log.debug("Got {} {} files: {}".format(len(files), label, [f.page.location.local_path() for f in files]))

    print_dataset_content(args.train_pcgts_files, 'training')
    print_dataset_content(args.validation_pcgts_files, 'validation')
    print_dataset_content(args.test_pcgts_files, 'testing')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    prediction_path = os.path.join(args.output, 'pred.json')
    model_path = os.path.join(args.output, 'best')

    if not args.global_args.skip_train and not args.global_args.do_not_use_model:
        fold_log.info("Starting training")

        settings = TrainSettings(
            n_iter=10000,
            n_classes=2,
            l_rate=1e-3,
            train_data=train_pcgts_dataset.to_page_segmentation_dataset(args.global_args.target_line_space, args.global_args.origin_line_space),
            validation_data=validation_pcgts_dataset.to_page_segmentation_dataset(args.global_args.target_line_space, args.global_args.origin_line_space),
            display=10,
            load=None,
            output=model_path,
            early_stopping_test_interval=100,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=8,
            data_augmentation=DefaultAugmenter(angle=(-2, 2), flip=(0.5, 0.5), contrast=0.8, brightness=20, scale=(-0.1, 0.1, -0.1, 0.1)) if args.global_args.data_augmentation else None,
            checkpoint_iter_delta=None,
            compute_baseline=True,
        )
        trainer = Trainer(settings)
        trainer.train()

    if not args.global_args.skip_prediction:
        fold_log.info("Starting evaluation")
        pred = create_staff_line_predictor(
            StaffLinesModelType.PIXEL_CLASSIFIER,
            StaffLinePredictorParameters(
                [model_path] if not args.global_args.do_not_use_model else None,
                # ["/home/cwick/Documents/Projects/ommr4all/modules/ommr4all-server/storage/Graduel/pc_staff_lines/model"],
                args.global_args.dataset_params,
                post_processing=args.global_args.post_processing,
                smooth_staff_lines=args.global_args.smooth_staff_lines,
                line_fit_distance=args.global_args.line_fit_distance,
            )
        )
        predictions = []
        for p in pred.predict(args.test_pcgts_files):
            predictions.append(
                EvaluationData(
                    p.line.operation.page.location.local_path(),
                    p.line.operation.music_lines,
                    p.music_lines,
                    p.line.operation.page_image.shape,
                )
            )

        with open(prediction_path, 'w') as f:
            json.dump({'predictions': [p.to_json() for p in predictions]}, f)
    else:
        with open(prediction_path, 'r') as f:
            predictions = [EvaluationData.from_json(p) for p in json.load(f)['predictions']]

    if not args.global_args.skip_eval:
        evaluator = StaffLineDetectionEvaluator()
        counts, prf1 = evaluator.evaluate(predictions)
        if counts.shape[0] > 0:
            at = PrettyTable()
            print(counts.shape)

            at.add_column("Type", ["Staff lines found", "Staff lines hit"])
            at.add_column("TP", counts[:, 0])
            at.add_column("FP", counts[:, 1])
            at.add_column("FN", counts[:, 2])
            at.add_column("Total", counts[:, 3])

            at.add_column("Precision", prf1[:, 0])
            at.add_column("Recall", prf1[:, 1])
            at.add_column("F1", prf1[:, 2])
            fold_log.debug(at.get_string())

            metrics = prf1
        else:
            logger.warning("Empty file without ground truth lines")
            metrics = None

    else:
        counts, metrics = None, None

    if not args.global_args.skip_cleanup:
        fold_log.info("Cleanup")
        shutil.rmtree(args.output)

    return counts, metrics


def flatten(data):
    out = []
    for d in data:
        out += d

    return out


def cross_fold(data, amount):
    folds = [data[i::amount] for i in range(amount)]
    return [(i, folds[i], flatten(folds[:i] + folds[i+1:])) for i in range(amount)]


class Experimenter:
    def __init__(self):
        pass

    def run(self, global_args: GlobalArgs):
        logger.info("Finding PcGts files with valid ground truth")
        all_pcgts, _ = dataset_by_locked_pages(1, [LockState('CreateStaffLines', True), LockState('Layout', True)])
        logger.info("Starting experiment with {} files".format(len(all_pcgts)))

        def prepare_single_fold(fold, train_val_files, test_files):
            _, val, train = cross_fold(train_val_files, 5)[0]
            return SingleDataArgs(fold, os.path.join(global_args.model_dir, "line_detection_{}".format(fold)), train, val, test_files, global_args)

        train_args = [
            prepare_single_fold(fold, train_val_files, test_files) for fold, test_files, train_val_files in cross_fold(all_pcgts, global_args.cross_folds)
        ]
        train_args = [train_args[fold] for fold in (global_args.single_folds if global_args.single_folds and len(global_args.single_folds) > 0 else range(global_args.cross_folds))]

        counts, metrics = zip(*map(run_single, train_args))

        logger.info("Total Result:")

        at = PrettyTable(["Fold", "TP", "FP", "FN", "Total", "Precision", "Recall", "F1"])
        for args, c, m in zip(train_args, counts, metrics):
            at.add_row([args.id] + list(c[0]) + list(m[0]))

        logger.info("\n\nStaff lines detected:\n" + at.get_string())

        at = PrettyTable(["Fold", "TP", "FP", "FN", "Total", "Precision", "Recall", "F1"])
        for args, c, m in zip(train_args, counts, metrics):
            at.add_row([args.id] + list(c[1]) + list(m[1]))

        logger.info("\n\nStaff lines hit:\n" + at.get_string())

        prec_rec_f1_list = metrics

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        at = PrettyTable()

        at.add_column("Type", ["Staff Lines Detected", "Staff lines hit"])

        at.add_column("Precision", prf1_mean[:, 0])
        at.add_column("+-", prf1_std[:, 0])
        at.add_column("Recall", prf1_mean[:, 1])
        at.add_column("+-", prf1_std[:, 1])
        at.add_column("F1", prf1_mean[:, 2])
        at.add_column("+-", prf1_std[:, 2])
        logger.info("\n\n" + at.get_string())


if __name__ == "__main__":
    import sys
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models_out/")
    parser.add_argument("--cross_folds", default=5, type=int)
    parser.add_argument("--single_folds", nargs="*", type=int, default=[])
    parser.add_argument("--target_line_space", default=10, type=int)
    parser.add_argument("--origin_line_space", default=None, type=int)
    parser.add_argument("--do_not_use_model", action="store_true", default=False)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_prediction", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--cleanup", action="store_true", default=False)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--post_processing", action="store_true")
    parser.add_argument("--smooth_staff_lines", type=int, default=0)
    parser.add_argument("--line_fit_distance", type=float, default=0)
    parser.add_argument("--n_train", default=-1, type=int)

    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--pad", default=[0], type=int, nargs="+")
    parser.add_argument("--full_page", action="store_true")
    parser.add_argument("--extract_region_only", action="store_true")
    parser.add_argument("--gt_line_thickness", default=3, type=int)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args = GlobalArgs(
        model_dir=args.model_dir,
        cross_folds=args.cross_folds,
        single_folds=args.single_folds,
        target_line_space=args.target_line_space,
        origin_line_space=args.origin_line_space,
        do_not_use_model=args.do_not_use_model,
        skip_train=args.skip_train,
        skip_prediction=args.skip_prediction,
        skip_eval=args.skip_eval,
        skip_cleanup=not args.cleanup,
        data_augmentation=args.data_augmentation,
        post_processing=args.post_processing,
        smooth_staff_lines=args.smooth_staff_lines,
        line_fit_distance=args.line_fit_distance,
        n_train=args.n_train,
        dataset_params=StaffLineDetectionDatasetParams(
            gt_required=True,
            full_page=args.full_page,
            gray=args.gray,
            pad=tuple(args.pad),
            extract_region_only=args.extract_region_only,
            gt_line_thickness=args.gt_line_thickness,
        ),
    )
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)
    experimenter = Experimenter()
    experimenter.run(args)

