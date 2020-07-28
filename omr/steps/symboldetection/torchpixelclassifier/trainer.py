import os

from omr.dataset import DatasetParams
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.imageoperations.music_line_operations import SymbolLabel
#from ocr4all_pixel_classifier.lib.trainer import Trainer, Loss, Monitor, Architecture
from omr.steps.symboldetection.torchpixelclassifier.meta import Meta
from segmentation.network import Network, TrainSettings
from segmentation.dataset import MemoryDataset
from omr.steps.symboldetection.torchpixelclassifier.callback import PCTorchTrainerCallback
class PCTorchTrainer(SymbolDetectionTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=20000,
            l_rate=1e-4,
            display=100,
            early_stopping_test_interval=500,
            early_stopping_max_keep=10,
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
        pc_callback = PCTorchTrainerCallback(callback) if callback else None
        if callback:
            callback.resolving_files()

        train_data = self.train_dataset.to_memory_dataset(callback)

        augmentation = self.settings.page_segmentation_torch_params.augmentation\
            if self.settings.page_segmentation_torch_params.data_augmentation else None
        train_data = MemoryDataset(df=train_data, transform=augmentation)
        #train_data.data = train_data.data * self.params.train_data_multiplier
        settings = TrainSettings(
            EPOCHS=max(1, self.settings.params.n_iter // len(train_data)),
            CLASSES=len(SymbolLabel),
            LEARNINGRATE_DECODER=self.params.l_rate,
            LEARNINGRATE_SEGHEAD=self.params.l_rate,
            LEARNINGRATE_ENCODER=1e-4,
            BATCH_ACCUMULATION=1,
            TRAIN_DATASET=train_data,
            VAL_DATASET=MemoryDataset(self.validation_dataset.to_memory_dataset(callback)),
            MODEL_PATH=None if not self.params.model_to_load() else self.params.model_to_load().local_file('model'),
            OUTPUT_PATH=self.settings.model.path + "/model",
            PROCESSES=self.params.processes,
            ENCODER=self.settings.page_segmentation_torch_params.encoder,
            ARCHITECTURE=self.settings.page_segmentation_torch_params.architecture
        )

        os.makedirs(os.path.dirname(self.settings.model.path), exist_ok=True)

        trainer = Network(settings)
        trainer.train(callback=pc_callback)


if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Graduel_Part_1')
    c = DatabaseBook('Graduel_Part_2')
    d = DatabaseBook('Graduel_Part_3')
    e = DatabaseBook('Pa_14819')

    train, val = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True)], datasets=[b, c, d, e])
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train,
        val,

    )
    trainer = PCTorchTrainer(settings)
    trainer.train(b)

