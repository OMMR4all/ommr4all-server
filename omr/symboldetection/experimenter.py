import logging
from omr.symboldetection.pixelclassifier.trainer import PCTrainer
from omr.dataset.datafiles import dataset_by_locked_pages
from omr.dataset.pcgtsdataset import PcGtsDataset, PcGts
from omr.symboldetection.evaluator import SymbolDetectionEvaluator, create_predictor, PredictorParameters, PredictorTypes, Counts, precision_recall_f1, AccCounts
from pagesegmentation.lib.trainer import Trainer, TrainSettings, TrainProgressCallback
from omr.imageoperations.music_line_operations import SymbolLabel
from typing import NamedTuple, List
import tempfile
import shutil
from prettytable import PrettyTable
import numpy as np

logger = logging.getLogger(__name__)


class SingleDataArgs(NamedTuple):
    id: int
    output: str
    train_pcgts_files: List[PcGts]
    validation_pcgts_files: List[PcGts]
    test_pcgts_files: List[PcGts]

    height: int = 80
    skip_train: bool = False
    skip_eval: bool = False
    skip_cleanup: bool = True


def run_single(args: SingleDataArgs):
    fold_log = logger.getChild("fold_{}".format(args.id))
    train_pcgts_dataset = PcGtsDataset(args.train_pcgts_files, gt_required=True, height=args.height)
    validation_pcgts_dataset = PcGtsDataset(args.validation_pcgts_files, gt_required=True, height=args.height)

    if not args.skip_train:
        fold_log.info("Starting training")
        settings = TrainSettings(
            n_iter=20000,
            n_classes=len(SymbolLabel),
            l_rate=1e-4,
            train_data=train_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            validation_data=validation_pcgts_dataset.to_music_line_page_segmentation_dataset(),
            load=None,
            display=100,
            output=args.output + "/best",
            early_stopping_test_interval=500,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=4,
        )

        trainer = Trainer(settings)
        trainer.train()

    if not args.skip_eval:
        fold_log.info("Starting evaluation")
        pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER,
                                PredictorParameters([args.output + '/best'], args.height))
        evaluator = SymbolDetectionEvaluator(pred)
        metrics, counts, acc_counts, acc_acc = evaluator.evaluate(args.test_pcgts_files)

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
        at.add_column("Type", ["Note all", "Note GC", "Note NS", "Clef type", "Accid type"])
        at.add_column("True", acc_counts[:, AccCounts.TRUE])
        at.add_column("False", acc_counts[:, AccCounts.FALSE])
        at.add_column("Total", acc_counts[:, AccCounts.TOTAL])
        at.add_column("Accuracy [%]", acc_acc[:, 0] * 100)

        fold_log.debug(at.get_string())
    else:
        prec_rec_f1 = None
        acc_acc = None


    if not args.skip_cleanup:
        fold_log.info("Cleanup")
        shutil.rmtree(args.output)

    return prec_rec_f1, acc_acc

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
        self.height = 80
        self.cross_folds = 5

    def run(self):
        logger.info("Finding PcGts files with valid ground truth")
        train_val_pcgts, test_pcgts = dataset_by_locked_pages(0.8, 'Symbol')
        logger.info("Starting experiment with {} training/val and {} test files".format(len(train_val_pcgts), len(test_pcgts)))
        logger.debug("Training/Validation files: {}".format([p.page.location.local_path() for p in train_val_pcgts]))
        logger.debug("Test files: {}".format([p.page.location.local_path() for p in test_pcgts]))

        train_args = [
            SingleDataArgs(fold, "/tmp/symbol_detection_{}".format(fold), train, val, test_pcgts, self.height)
            for fold, val, train in cross_fold(train_val_pcgts, self.cross_folds)
        ]

        results = list(map(run_single, train_args))

        logger.info("Total Result:")

        prec_rec_f1_list = [r for r, _ in results]
        acc_counts_list = [r for _, r in results]

        prf1_mean = np.mean(prec_rec_f1_list, axis=0)
        prf1_std = np.std(prec_rec_f1_list, axis=0)

        acc_mean = np.mean(acc_counts_list, axis=0)
        acc_std = np.std(acc_counts_list, axis=0)

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
        at.add_column("Type", ["Note all", "Note GC", "Note NS", "Clef type", "Accid type"])
        at.add_column("Accuracy [%]", acc_mean[:, 0] * 100)
        at.add_column("+- [%]", acc_std[:, 0] * 100)

        logger.info("\n" + at.get_string())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)
    experimenter = Experimenter()
    experimenter.run()

