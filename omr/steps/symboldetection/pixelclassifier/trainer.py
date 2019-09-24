import os

from omr.dataset import DatasetParams

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Type
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.imageoperations.music_line_operations import SymbolLabel
from pagesegmentation.lib.trainer import Trainer, TrainSettings, Loss, Monitor, Architecture
from omr.steps.symboldetection.pixelclassifier.meta import Meta
from omr.adapters.pagesegmentation.callback import PCTrainerCallback


class PCTrainer(AlgorithmTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=10000,
            l_rate=1e-4,
            display=100,
            early_stopping_test_interval=500,
            early_stopping_max_keep=5,
            processes=1,
        )

    @staticmethod
    def default_dataset_params() -> DatasetParams:
        return DatasetParams(
            pad=[0, 10, 0, 40],
            dewarp=False,
            center=False,
            staff_lines_only=True,
            cut_region=False,
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.pad_power_of_2 = True

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        pc_callback = PCTrainerCallback(callback) if callback else None
        if callback:
            callback.resolving_files()

        train_data = self.train_dataset.to_page_segmentation_dataset(callback)
        settings = TrainSettings(
            n_epoch=max(1, self.settings.params.n_iter // len(train_data)),
            n_classes=len(SymbolLabel),
            l_rate=self.params.l_rate,
            train_data=train_data,
            validation_data=self.validation_dataset.to_page_segmentation_dataset(callback),
            load=None if not self.params.model_to_load() else self.params.model_to_load().local_file('model'),
            display=self.params.display,
            output_dir=self.settings.model.path,
            model_name='model',
            early_stopping_max_l_rate_drops=self.params.early_stopping_max_keep,
            threads=self.params.processes,
            compute_baseline=True,
            data_augmentation=self.settings.page_segmentation_params.data_augmentation,
            data_augmentation_settings=self.settings.page_segmentation_params.augmentation_settings,
            loss=Loss.CATEGORICAL_CROSSENTROPY,
            monitor=Monitor.VAL_ACCURACY,
            architecture=Architecture.FCN_SKIP,
        )

        os.makedirs(os.path.dirname(settings.output_dir), exist_ok=True)

        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Graduel_Fully_Annotated')
    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True)], datasets=[b])
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train,
        val,

    )
    trainer = PCTrainer(settings)
    trainer.train(b)

