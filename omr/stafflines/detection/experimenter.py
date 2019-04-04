import logging
from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.performance import LockState
from pagesegmentation.lib.trainer import TrainSettings, Trainer
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
from omr.dataset.datafiles import dataset_by_locked_pages, generate_dataset, GeneratedData
from omr.stafflines.detection.dataset import PCDataset, StaffLineDetectionDatasetParams
from omr.stafflines.detection.predictor import create_staff_line_predictor, StaffLinesModelType, StaffLinePredictorParameters
from omr.stafflines.detection.evaluator import StaffLineDetectionEvaluator, EvaluationData, EvaluationParams
from typing import NamedTuple, List, Optional
import os
import shutil
from prettytable import PrettyTable
import numpy as np
import json

logger = logging.getLogger(__name__)


class GlobalArgs(NamedTuple):
    magic_prefix: Optional[str]
    model_dir: str
    cross_folds: int
    single_folds: Optional[List[int]]
    n_train: int
    n_iter: int
    val_amount: float
    pretrained_model: Optional[str]

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

    # evaluation parameters
    evaluation_params: EvaluationParams


class SingleDataArgs(NamedTuple):
    id: int
    output: str
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: Optional[List[PcGts]]
    test_pcgts_files: List[PcGts]

    global_args: GlobalArgs


def run_single(args: SingleDataArgs):
    fold_log = logger.getChild("fold_{}".format(args.id))
    train_pcgts_dataset = PCDataset(
        args.train_pcgts_files,
        args.global_args.dataset_params)
    validation_pcgts_dataset = PCDataset(args.validation_pcgts_files, args.global_args.dataset_params) if args.validation_pcgts_files else None

    def print_dataset_content(files: List[PcGts], label: str):
        fold_log.debug("Got {} {} files: {}".format(len(files), label, [f.page.location.local_path() for f in files]))

    print_dataset_content(args.train_pcgts_files, 'training')
    if args.validation_pcgts_files:
        print_dataset_content(args.validation_pcgts_files, 'validation')
    else:
        fold_log.debug("No validation data. Using training data instead")

    print_dataset_content(args.test_pcgts_files, 'testing')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    prediction_path = os.path.join(args.output, 'pred.json')
    model_path = os.path.join(args.output, 'best')

    if not args.global_args.skip_train and not args.global_args.do_not_use_model:
        fold_log.info("Starting training")
        train_data = train_pcgts_dataset.to_page_segmentation_dataset(args.global_args.target_line_space, args.global_args.origin_line_space)
        validation_data = validation_pcgts_dataset.to_page_segmentation_dataset(args.global_args.target_line_space, args.global_args.origin_line_space) if validation_pcgts_dataset else train_data

        settings = TrainSettings(
            n_iter=args.global_args.n_iter,
            n_classes=2,
            l_rate=1e-3,
            train_data=train_data,
            validation_data=validation_data,
            display=10,
            load=args.global_args.pretrained_model,
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
        evaluator = StaffLineDetectionEvaluator(args.global_args.evaluation_params)
        counts, prf1 = evaluator.evaluate(predictions)
        if counts.shape[0] > 0:
            at = PrettyTable()
            print(counts.shape)

            at.add_column("Type", ["Staff lines found", "Staff lines hit", "Staves found", "Staff lines hit"])
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


class Experimenter:
    def __init__(self):
        pass

    def run(self, global_args: GlobalArgs, train_books: Optional[List[str]], test_books: Optional[List[str]], train_books_extend: Optional[List[str]]):
        train_args = generate_dataset(
            lock_states=[LockState('StaffLines', True), LockState('Layout', True)],
            n_train=global_args.n_train,
            val_amount=global_args.val_amount,
            cross_folds=global_args.cross_folds,
            single_folds=global_args.single_folds,
            train_books=train_books,
            test_books=test_books,
            train_books_extend=train_books_extend,
        )

        train_args = [SingleDataArgs(gd.fold,
                                     os.path.join(global_args.model_dir, 'line_detection_{}'.format(gd.fold)),
                                     gd.train_pcgts_files, gd.validation_pcgts_files,
                                     gd.test_pcgts_files,
                                     global_args) for gd in train_args]

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

        at.add_column("Type", ["Staff Lines Detected", "Staff lines hit", "Staves found", "Staff lines hit"])

        at.add_column("Precision", prf1_mean[:, 0])
        at.add_column("+-", prf1_std[:, 0])
        at.add_column("Recall", prf1_mean[:, 1])
        at.add_column("+-", prf1_std[:, 1])
        at.add_column("F1", prf1_mean[:, 2])
        at.add_column("+-", prf1_std[:, 2])
        logger.info("\n\n" + at.get_string())

        if global_args.magic_prefix:
            all_values = np.array(sum([[prf1_mean[:, i], prf1_std[:, i]] for i in range(3)], [])).transpose().reshape(-1)
            print("{}{}".format(global_args.magic_prefix, ','.join(map(str, all_values))))


if __name__ == "__main__":
    import sys
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--magic_prefix', default='EXPERIMENT_OUT=')
    parser.add_argument("--train", default=None, nargs="+")
    parser.add_argument("--test", default=None, nargs="+")
    parser.add_argument("--train_extend", default=None, nargs="+")
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
    parser.add_argument("--n_iter", default=10000, type=int)
    parser.add_argument("--val_amount", default=0.2, type=float)
    parser.add_argument("--pretrained_model", default=None, type=str)

    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--pad", default=[0], type=int, nargs="+")
    parser.add_argument("--full_page", action="store_true")
    parser.add_argument("--extract_region_only", action="store_true")
    parser.add_argument("--gt_line_thickness", default=3, type=int)

    # evaluation parameters
    parser.add_argument("--staff_line_found_distance", default=5, type=int)
    parser.add_argument("--line_hit_overlap_threshold", default=0.5, type=float)
    parser.add_argument("--staff_n_lines_threshold", default=2, type=int)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global_args = GlobalArgs(
        magic_prefix=args.magic_prefix,
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
        n_iter=args.n_iter,
        val_amount=args.val_amount,
        pretrained_model=args.pretrained_model,
        dataset_params=StaffLineDetectionDatasetParams(
            gt_required=True,
            full_page=args.full_page,
            gray=args.gray,
            pad=tuple(args.pad),
            extract_region_only=args.extract_region_only,
            gt_line_thickness=args.gt_line_thickness,
        ),
        evaluation_params=EvaluationParams(
            staff_line_found_distance=args.staff_line_found_distance,
            line_hit_overlap_threshold=args.line_hit_overlap_threshold,
            staff_n_lines_threshold=args.staff_n_lines_threshold,
        )
    )
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)
    experimenter = Experimenter()
    experimenter.run(global_args, args.train, args.test, args.train_extend)

