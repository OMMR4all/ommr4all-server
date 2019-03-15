from omr.stafflines.detection.trainer import StaffLinesTrainer
from omr.dataset.datafiles import dataset_by_locked_pages, EmptyDataSetException, LockState
from database import DatabaseBook
import os
import logging
from omr.stafflines.detection.pixelclassifier.dataset import PCDataset
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
import omr.stafflines.detection.pixelclassifier.settings as pc_settings

logger = logging.getLogger(__name__)


class BasicStaffLinesTrainer(StaffLinesTrainer):
    def __init__(self, b: DatabaseBook):
        super().__init__(b)

    def train(self):
        train, val = dataset_by_locked_pages(0.8, [LockState('CreateStaffLines', True)])
        if len(train) == 0 or len(val) == 0:
            raise EmptyDataSetException()

        from pagesegmentation.lib.trainer import TrainSettings, Trainer

        settings = TrainSettings(
            n_iter=1000,
            n_classes=2,
            l_rate=1e-3,
            train_data=PCDataset(train, True).to_page_segmentation_dataset(target_staff_line_distance=10),
            validation_data=PCDataset(val, True).to_page_segmentation_dataset(target_staff_line_distance=10),
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
        trainer.train()


if __name__=="__main__":
    trainer = BasicStaffLinesTrainer(DatabaseBook('Graduel'))
    trainer.train()
