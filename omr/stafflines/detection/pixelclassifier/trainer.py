from omr.stafflines.detection.trainer import StaffLinesTrainer, Optional, StaffLinesDetectionTrainerCallback
from omr.dataset.datafiles import dataset_by_locked_pages, EmptyDataSetException, LockState
from database import DatabaseBook
import os
import logging
from omr.stafflines.detection.dataset import PCDataset, StaffLineDetectionDatasetParams
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
import omr.stafflines.detection.pixelclassifier.settings as pc_settings
from pagesegmentation.lib.trainer import TrainSettings, Trainer, TrainProgressCallback

logger = logging.getLogger(__name__)


class PCTrainerCallback(TrainProgressCallback):
    def __init__(self, callback: StaffLinesDetectionTrainerCallback):
        super().__init__()
        self.callback = callback

    def init(self, total_iters, early_stopping_iters):
        self.callback.init(total_iters, early_stopping_iters)

    def next_iteration(self, iter: int, loss: float, acc: float, fgpa: float):
        self.callback.next_iteration(iter, loss, acc)

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        self.callback.next_best_model(best_iter, best_acc, best_iters)

    def early_stopping(self):
        self.callback.early_stopping()


class BasicStaffLinesTrainer(StaffLinesTrainer):
    def __init__(self, b: DatabaseBook):
        super().__init__(b)

    def train(self, callback: Optional[StaffLinesDetectionTrainerCallback]=None):
        pc_callback = PCTrainerCallback(callback) if callback else None

        train, val = dataset_by_locked_pages(0.8, [LockState('StaffLines', True)])
        if len(train) == 0 or len(val) == 0:
            raise EmptyDataSetException()

        settings = TrainSettings(
            n_iter=1000,
            n_classes=2,
            l_rate=1e-3,
            train_data=PCDataset(train, StaffLineDetectionDatasetParams(True)).to_page_segmentation_dataset(target_staff_line_distance=10),
            validation_data=PCDataset(val, StaffLineDetectionDatasetParams(True)).to_page_segmentation_dataset(target_staff_line_distance=10),
            load=None,
            display=10,
            output=self.book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name)),
            early_stopping_test_interval=50,
            early_stopping_max_keep=5,
            early_stopping_on_accuracy=True,
            threads=8,
            data_augmentation=DefaultAugmenter(angle=(-2, 2), flip=(0.5, 0.5), contrast=0.8, brightness=20, scale=(-0.2, 0.2, -0.2, 0.2)),
            checkpoint_iter_delta=None,
        )
        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__=="__main__":
    trainer = BasicStaffLinesTrainer(DatabaseBook('Graduel'))
    trainer.train()
