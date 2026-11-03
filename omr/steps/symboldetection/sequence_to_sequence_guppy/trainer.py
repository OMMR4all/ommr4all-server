import csv
import os
import json
import shutil
import tempfile

import loguru
import torch
from PIL import Image
import numpy as np
from omr.dataset.datastructs import CalamariCodec
from omr.dataset.sample_store import write_line_samples
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional, Tuple, List
from database import DatabaseBook
from omr.dataset import DatasetParams
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.steps.symboldetection.sequence_to_sequence_guppy.meta import Meta

from database.file_formats.performance.pageprogress import Locks

this_dir = os.path.dirname(os.path.realpath(__file__))


class OMRTrainer(SymbolDetectionTrainer):
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
        params.dewarp = True
        params.center = False # Also inserts padding
        params.staff_lines_only = True
        params.pad_power_of_2 = False
        params.calamari_codec = CalamariCodec()
        with open(os.path.join(this_dir, 'default_codec.json'), 'r') as f:
            params.calamari_codec = CalamariCodec.from_dict(json.load(f))

    def __init__(self, params: AlgorithmTrainerSettings):
        super().__init__(params)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        if callback:
            callback.resolving_files()

        from ommr4all.settings import BASE_DIR
        from omr.steps.symboldetection.sequence_to_sequence_guppy.arch import train_model

        with tempfile.TemporaryDirectory() as dirpath:
            loguru.logger.info(f"Creating temporary train directory at {dirpath}")

            # the samples are written one by one as they are built: guppyocr reads them from
            # disk anyway, so the whole dataset never has to fit into memory
            n_train = write_line_samples(
                os.path.join(dirpath, "train"),
                self.train_dataset.iter_guppy_symbol_line_samples(train=True, callback=callback,
                                                                  only_with_gt=True))
            n_val = write_line_samples(
                os.path.join(dirpath, "test"),
                self.validation_dataset.iter_guppy_symbol_line_samples(train=True, callback=callback,
                                                                       only_with_gt=True))
            loguru.logger.info(f"Training samples: {n_train}, validation samples: {n_val}")
            if n_val == 0:
                # no validation data: validate on the training data, as before
                shutil.rmtree(os.path.join(dirpath, "test"))
                shutil.copytree(os.path.join(dirpath, "train"), os.path.join(dirpath, "test"))
            from guppyocr.train_calamares import TrainingOpts
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print(self.params.model_to_load().local_file('model_best.pth'))
            training_opts = TrainingOpts(
                output=self.settings.model.path,
                dataset=os.path.join(dirpath),
                #val_dataset=os.path.join(dirpath, "test"),
                model='' if not self.params.model_to_load() else self.params.model_to_load().local_file('model_best.pth'),
                test_loaded_model=False,
                reader="plain",
                arch="symbolcrnn",
                gpu=True,
                worker=2,
                epoch=self.params.n_epoch,
                grad_clip=False,
                img_height=112,
                img_width=800,
                augment=self.settings.page_segmentation_torch_params.data_augmentation
            )
            train_model(training_opts)


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel_Part_1')
    c = DatabaseBook('Graduel_Part_2')
    d = DatabaseBook('Graduel_Part_3')
    f = DatabaseBook('Pa_14819')

    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [b, c, d, f])
    dataset_params = DatasetParams(
        gt_required=True,
        height=128,
        dewarp=True,
        cut_region=False,
        pad=[0, 20, 0, 20],
        center=False,
        staff_lines_only=True,
        masks_as_input=False,
    )
    train_settings = AlgorithmTrainerSettings(
        dataset_params=dataset_params,
        train_data=train_pcgts,
        validation_data=val_pcgts,
        params=AlgorithmTrainerParams(
            l_rate=1e-3,
        )
    )
    trainer = OMRTrainer(train_settings)
    trainer.train(b)



