import os

from omr.dataset import DatasetParams
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Type
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmTrainer, TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.imageoperations.music_line_operations import SymbolLabel
from ocr4all_pixel_classifier.lib.trainer import Trainer, TrainSettings, Loss, Monitor, Architecture
from omr.steps.symboldetection.centernet.meta import Meta
from omr.adapters.pagesegmentation.callback import PCTrainerCallback

class Config:
    def as_dict(self):
        return vars(self)

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self)


class CenterNetTrainer(SymbolDetectionTrainer):
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
        from centernet.dataset import MemoryCenterNet
        from centernet.train import train
        import pandas as pd
        config = Config()
        config.id = 38
        config.epochs = 125
        config.batch_size = 4  # TODO
        config.lr = 5e-5  # 1e-4
        config.weight_decay = 1e-4
        config.weight = None
        config.warmup = 0.03
        config.accumulation_step = 1
        config.num_folds = 10
        config.num_workers = 0
        config.p_clf = 0.6
        config.p_seg = 0.2
        config.pin = False
        config.slug = 'r50'  # TODO
        config.device = 'cuda'
        config.apex = False  # TODO
        config.subset = -1  # TODO
        config.n_keep = 1
        config.is_kernel = False
        config.cache = False
        config.p_letter = 0.1
        config.p_class = 0.1
        config.add_val = True
        config.logdir = self.settings.model.path
        config.w_h_ratio = 0.2
        config.fold = 1
        pc_callback = PCTrainerCallback(callback) if callback else None
        if callback:
            callback.resolving_files()

        train_data = self.train_dataset.to_centernet_dataset(callback)
        images, bboxes = train_data
        ratio = int(0.8 * len(bboxes))
        df_train = pd.DataFrame(data={'images': images[:ratio], 'bbox': bboxes[:ratio]})
        df_val = pd.DataFrame(data={'images': images[:ratio], 'bbox': bboxes[:ratio]})
        train(df_train, df_val, config, MemoryCenterNet)


if __name__ == '__main__':
    from omr.dataset import DatasetParams
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    b = DatabaseBook('Graduel_Part_1')
    c = DatabaseBook('Graduel_Part_2')
    d = DatabaseBook('Graduel_Part_3')

    train, val = dataset_by_locked_pages(1, [LockState(Locks.STAFF_LINES, True)], datasets=[b, c, d])
    settings = AlgorithmTrainerSettings(
        DatasetParams(),
        train,
        val,

    )
    trainer = CenterNetTrainer(settings)
    trainer.train(b)
