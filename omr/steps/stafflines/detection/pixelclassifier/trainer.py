from omr.dataset.datafiles import dataset_by_locked_pages, EmptyDataSetException, LockState
from database import DatabaseBook
import logging
from omr.steps.stafflines.detection.dataset import PCDataset, DatasetParams
from omr.adapters.pagesegmentation.callback import PCTrainerCallback
from pagesegmentation.lib.data_augmenter import DefaultAugmenter
from pagesegmentation.lib.trainer import TrainSettings, Trainer
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings, Dataset
from omr.steps.stafflines.detection.pixelclassifier.meta import Meta, AlgorithmMeta
from typing import Optional, Type
import os

logger = logging.getLogger(__name__)


class BasicStaffLinesTrainer(AlgorithmTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=1000,
            l_rate=1e-3,
            display=10,
            early_stopping_max_keep=5,
            early_stopping_test_interval=50,
        )

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        pc_callback = PCTrainerCallback(callback) if callback else None

        if callback:
            callback.resolving_files()

        settings = TrainSettings(
            n_iter=self.params.n_iter,
            n_classes=2,
            l_rate=self.params.l_rate,
            train_data=self.train_dataset.to_page_segmentation_dataset(callback=callback),
            validation_data=self.validation_dataset.to_page_segmentation_dataset(callback=callback),
            load=None if not self.params.model_to_load() else self.params.model_to_load().local_file('model'),
            display=self.params.display,
            output=self.settings.model.local_file('model'),
            early_stopping_test_interval=self.params.early_stopping_test_interval,
            early_stopping_max_keep=self.params.early_stopping_max_keep,
            early_stopping_on_accuracy=True,
            threads=8,
            data_augmentation=DefaultAugmenter(angle=(-2, 2), flip=(0.5, 0.5), contrast=0.2, brightness=5),
            checkpoint_iter_delta=None,
        )
        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == "__main__":
    from database import DatabaseBook
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    book = DatabaseBook('Graduel_Fully_Annotated')
    train, val = dataset_by_locked_pages(0.8, [LockState('StaffLines', True)], datasets=[book])
    trainer = BasicStaffLinesTrainer(AlgorithmTrainerSettings(
        dataset_params=DatasetParams(),
        train_data=train,
        validation_data=val,
    ))
    trainer.train(book)
