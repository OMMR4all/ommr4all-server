from typing import Optional, Type
from database import DatabaseBook
import os
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings, Dataset, Model
from omr.imageoperations.music_line_operations import SymbolLabel
from pagesegmentation.lib.trainer import Trainer, TrainSettings
from omr.steps.symboldetection.pixelclassifier.meta import Meta
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
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

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        pc_callback = PCTrainerCallback(callback) if callback else None
        settings = TrainSettings(
            n_iter=self.settings.params.n_iter,
            n_classes=len(SymbolLabel),
            l_rate=self.params.l_rate,
            train_data=self.train_dataset.to_page_segmentation_dataset(callback),
            validation_data=self.validation_dataset.to_page_segmentation_dataset(callback),
            load=None if not self.params.model_to_load() else self.params.model_to_load().local_file('model'),
            display=self.params.display,
            output=self.settings.model.local_file('model'),
            early_stopping_test_interval=self.params.early_stopping_test_interval,
            early_stopping_max_keep=self.params.early_stopping_max_keep,
            early_stopping_on_accuracy=True,
            threads=self.params.processes,
            checkpoint_iter_delta=None,
            compute_baseline=True,
            data_augmentation=self.settings.page_segmentation_params.data_augmenter,
        )

        if not os.path.exists(os.path.dirname(settings.output)):
            os.makedirs(os.path.dirname(settings.output))

        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Graduel_Fully_Annotated')
    train, val = dataset_by_locked_pages(0.8, [LockState('StaffLines', True)], datasets=[b])
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train,
        val,

    )
    trainer = PCTrainer(settings)
    trainer.train(b)

