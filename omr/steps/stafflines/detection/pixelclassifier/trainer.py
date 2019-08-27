import os
if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from database.file_formats.performance.pageprogress import Locks
from database import DatabaseBook
import logging
from omr.steps.stafflines.detection.dataset import DatasetParams
from omr.adapters.pagesegmentation.callback import PCTrainerCallback
from pagesegmentation.lib.trainer import TrainSettings, Trainer
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.steps.stafflines.detection.pixelclassifier.meta import Meta
from typing import Optional

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

        train_data = self.train_dataset.to_page_segmentation_dataset(callback)
        settings = TrainSettings(
            n_epoch=max(1, self.settings.params.n_iter // len(train_data)),
            n_classes=2,
            l_rate=self.params.l_rate,
            train_data=train_data,
            validation_data=self.validation_dataset.to_page_segmentation_dataset(callback=callback),
            load=None if not self.params.model_to_load() else self.params.model_to_load().local_file('model'),
            display=self.params.display,
            output_dir=self.settings.model.path,
            best_model_name='model',
            early_stopping_max_l_rate_drops=self.params.early_stopping_max_keep,
            threads=self.params.processes,
            data_augmentation=False,
        )
        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == "__main__":
    from database import DatabaseBook
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    book = DatabaseBook('Graduel_Fully_Annotated')
    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True)], datasets=[book])
    trainer = BasicStaffLinesTrainer(AlgorithmTrainerSettings(
        dataset_params=DatasetParams(),
        train_data=train,
        validation_data=val,
    ))
    trainer.train(book)
