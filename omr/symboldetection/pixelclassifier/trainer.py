from typing import Optional
from database.file_formats import PcGts
from database import DatabaseBook
import os
from omr.imageoperations.music_line_operations import SymbolLabel
from pagesegmentation.lib.trainer import Trainer, TrainSettings, TrainProgressCallback
from omr.symboldetection.trainer import SymbolDetectionTrainerCallback, SymbolDetectionTrainerBase, SymbolDetectionTrainerParams
import omr.symboldetection.pixelclassifier.settings as pc_settings


class PCTrainerCallback(TrainProgressCallback):
    def __init__(self, callback: SymbolDetectionTrainerCallback):
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


class PCTrainer(SymbolDetectionTrainerBase):
    def __init__(self, params: SymbolDetectionTrainerParams):
        super().__init__(params)

    def run(self, model_for_book: Optional[DatabaseBook]=None, callback: Optional[SymbolDetectionTrainerCallback]=None):
        pc_callback = PCTrainerCallback(callback) if callback else None

        settings = TrainSettings(
            n_iter=self.params.n_iter if self.params.n_iter > 0 else 10000,
            n_classes=len(SymbolLabel),
            l_rate=self.params.l_rate if self.params.l_rate > 0 else 1e-4,
            train_data=self.train_dataset.to_music_line_page_segmentation_dataset(),
            validation_data=self.validation_dataset.to_music_line_page_segmentation_dataset(),
            load=self.params.load,
            display=self.params.display,
            output=self.params.output if self.params.output else model_for_book.local_path(os.path.join(pc_settings.model_dir, pc_settings.model_name)),
            early_stopping_test_interval=self.params.early_stopping_test_interval if self.params.early_stopping_test_interval >= 0 else 500,
            early_stopping_max_keep=self.params.early_stopping_max_keep if self.params.early_stopping_max_keep >= 0 else 5,
            early_stopping_on_accuracy=True,
            threads=self.params.processes if self.params.processes > 0 else 1,
            checkpoint_iter_delta=None,
            compute_baseline=True,
            data_augmentation=self.params.page_segmentation_params.data_augmenter,
        )

        if not os.path.exists(os.path.dirname(settings.output)):
            os.makedirs(os.path.dirname(settings.output))

        trainer = Trainer(settings)
        trainer.train(callback=pc_callback)


if __name__ == '__main__':
    from omr.symboldetection.dataset import SymbolDetectionDatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Graduel')
    train, val = dataset_by_locked_pages(0.8, [LockState('StaffLines', True)], datasets=[b])
    params = SymbolDetectionTrainerParams(
        SymbolDetectionDatasetParams(),
        train,
        val,
    )
    trainer = PCTrainer(params)
    trainer.run(b)

