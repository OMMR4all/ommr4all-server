import os
import tempfile
from pathlib import Path

from omr.dataset import DatasetParams
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Tuple, Dict
from database import DatabaseBook

from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.steps.symboldetection.yolo_detector.meta import Meta

import yaml


class YoloTrainer(SymbolDetectionTrainer):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=10_000,
            l_rate=1e-3,
            display=100,
            early_stopping_test_interval=1000,
            early_stopping_max_keep=5,
            processes=1,
        )

    @staticmethod
    def default_dataset_params() -> DatasetParams:
        return DatasetParams(
            pad=[0, 10, 0, 40],
            cut_region=False,
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.dewarp = False
        params.center = False  # Also inserts padding
        params.staff_lines_only = True
        params.pad_power_of_2 = False

    def __init__(self, params: AlgorithmTrainerSettings):
        super().__init__(params)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        from omr.imageoperations.music_line_operations import SymbolLabel

        if callback:
            callback.resolving_files()

        def make_yaml(train: Path, val: Path, yaml_fname) -> Dict[int, str]:
            root = train.parent
            lookup_dict = {}
            for i in enumerate(SymbolLabel):
                if i[0] == 0:
                    continue

                lookup_dict[i[0] - 1] = i[1].name.lower()
            yaml_dict = {
                "path": str(root.absolute()),
                "train": str(train.name),
                "val": str(val.name),
                "names": lookup_dict,
            }
            with open(yaml_fname, 'w') as f:
                yaml.dump(yaml_dict, f)
            return lookup_dict

        with tempfile.TemporaryDirectory() as dirpath:
            os.mkdir(os.path.join(dirpath, "train"))
            os.mkdir(os.path.join(dirpath, "val"))
            train_path = Path(os.path.join(dirpath, "train"))
            val_path = Path(os.path.join(dirpath, "val"))
            yaml_path = Path(os.path.join(dirpath, "data.yaml"))
            look_up = make_yaml(train_path, val_path, yaml_path)
            train_dataset = self.train_dataset.to_yolo_symbol_dataset(train=True, train_path=train_path,
                                                                      callback=callback)
            val_dataset = self.validation_dataset.to_yolo_symbol_dataset(train=False, train_path=val_path,
                                                                         callback=callback)
            from ultralytics import YOLO

            # Load the model.
            model = YOLO("yolo11n.pt")
            os.makedirs(os.path.dirname(self.settings.model.path), exist_ok=True)

            # Training.
            results = model.train(
                data=yaml_path,
                #imgsz=(160, 2560),
                epochs=self.settings.params.n_epoch,
                #batch=4,
                name='model',
                augment=self.settings.page_segmentation_torch_params.data_augmentation,
                save=True,
                project=self.settings.model.path
            )
        # val_dataset = self.validation_dataset.to_nautilus_dataset(train=False, callback=callback)


if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(1)
    np.random.seed(1)
    #b = DatabaseBook('Graduel_Syn')

    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    b = DatabaseBook('Graduel_Part_1_gt')
    c = DatabaseBook('Graduel_Part_2_gt')
    d = DatabaseBook('Graduel_Part_3_gt')
    # e = DatabaseBook('Pa_14819_gt')
    # f = DatabaseBook('Assisi')
    # g = DatabaseBook('Cai_72')
    # h = DatabaseBook('pa_904')
    #i = DatabaseBook('mul_2_rsync_gt2')
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True),
                                                           LockState(Locks.LAYOUT, True)], True, [b, c, d])

    dataset_params = DatasetParams(
        pad=[0, 10, 0, 40],
        dewarp=False,
        center=False,
        staff_lines_only=True,
        cut_region=False,
    )
    train_settings = AlgorithmTrainerSettings(
        dataset_params=dataset_params,
        train_data=train_pcgts,
        validation_data=val_pcgts,
        params=AlgorithmTrainerParams(
            l_rate=1e-3,
        )
    )
    trainer = YoloTrainer(train_settings)
    trainer.train(b)
