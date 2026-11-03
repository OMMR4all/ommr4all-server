import csv
import shutil
from typing import Optional, Type, Dict, List, Tuple

import numpy as np
import torch
from guppyocr.train_calamares import train_model

from database.file_formats.pcgts import BlockType
from database.file_formats.performance.pageprogress import Locks
from omr.dataset import DatasetParams, LyricsNormalization
from omr.dataset.sample_store import write_line_samples
from omr.steps.symboldetection.sequencetosequence.params import CalamariParams
from database import DatabaseBook
import os
from omr.steps.algorithm import AlgorithmTrainerParams, AlgorithmTrainerSettings, TrainerCallback, AlgorithmMeta
from omr.steps.text.trainer import TextTrainerBase
import tempfile
from PIL import Image
import loguru
import yaml
import pandas as pd
import uuid


class CalamariTrainerCallback:
    def __init__(self, cb: TrainerCallback):
        self.cb = cb

    def display(self, train_cer, train_loss, train_dt, iter, steps_per_epoch, display_epochs,
                example_pred, example_gt):
        self.cb.next_iteration(iter, train_loss, 1 - train_cer)

    def early_stopping(self, eval_cer, n_total, n_best, iter):
        self.cb.next_best_model(iter, eval_cer, n_best - 1)


class PytorchGuppyyTrainer(TextTrainerBase):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.guppy.meta import Meta
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_iter=100_000,
            l_rate=1e-3,
            display=100,
            early_stopping_test_interval=1000,
            early_stopping_max_keep=5,
            processes=1,
            data_augmentation_factor=20,
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.height = 64
        params.text_image_color_type = "color"
        params.pad = [5, 5, 5, 5]
        params.text_types = [BlockType.LYRICS]

        # params.lyrics_normalization = params.lyrics_normalization.lyrics_normalization.WORDS

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        from ommr4all.settings import BASE_DIR

        from guppyocr.train_calamares import train_model

        from omr.steps.text.guppy.progress import report_training_progress

        n_epoch = max(int(self.settings.params.n_epoch or 0), 1)
        # guppyocr breaks once epochs_without_improvement >= early_stopping, so a
        # non-positive value would end the run after the very first epoch
        early_stopping = int(self.settings.params.early_stopping_max_keep or 0)
        if early_stopping <= 0:
            early_stopping = self.default_params().early_stopping_max_keep
        if callback:
            callback.resolving_files()
            # guppyocr's loop counts epochs, not iterations, so rescale the progress
            # (AlgorithmTrainer.train initialises it with n_iter). Must be > 0: the task
            # runner divides by total_iters without a guard.
            callback.init(n_epoch, early_stopping)

        with tempfile.TemporaryDirectory() as dirpath:
            loguru.logger.info(f"Creating temporary train directory at {dirpath}")

            # the samples are written one by one as they are built: guppyocr reads them from
            # disk anyway, so the whole dataset never has to fit into memory
            n_train = write_line_samples(
                os.path.join(dirpath, "train"),
                self.train_dataset.iter_text_line_nautilus_samples(train=True, callback=callback,
                                                                   only_with_gt=True),
                unique_names=True)
            n_val = write_line_samples(
                os.path.join(dirpath, "test"),
                self.validation_dataset.iter_text_line_nautilus_samples(train=True, callback=callback,
                                                                        only_with_gt=True),
                unique_names=True)
            loguru.logger.info(f"Training samples: {n_train}, validation samples: {n_val}")
            if n_val == 0:
                # no validation data: validate on the training data, as before
                shutil.rmtree(os.path.join(dirpath, "test"))
                shutil.copytree(os.path.join(dirpath, "train"), os.path.join(dirpath, "test"))
            from guppyocr.train_calamares import TrainingOpts
            training_opts = TrainingOpts(
                output=self.settings.model.path,
                dataset=os.path.join(dirpath),
                #val_dataset=os.path.join(dirpath, "test"),
                model='' if not self.params.model_to_load() else self.params.model_to_load().local_file('model_best.pth'),
                test_loaded_model=False,
                reader="plain",
                arch="crnn",
                gpu=True,
                worker=2,
                epoch=n_epoch,
                # without this guppyocr's default of 1000000 applies and training never
                # stops early, however long the validation CER stays flat
                early_stopping=early_stopping,
                grad_clip=False,
                augment=True,
            )
            with report_training_progress(callback, n_epoch):
                train_model(training_opts)



if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(1)
    np.random.seed(1)

    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    import django

    django.setup()
    #b = DatabaseBook('Graduel_Part_1_gt')
    #c = DatabaseBook('Graduel_Part_2_gt')
    #d = DatabaseBook('Graduel_Part_3_gt')
    #e = DatabaseBook('Pa_14819_gt')
    #f = DatabaseBook('Assisi')
    #g = DatabaseBook('Cai_72')
    # h = DatabaseBook('mul_2_rsync_gt')
    #hg = DatabaseBook('')
    e = DatabaseBook('Geesebook2_30_09')
    f = DatabaseBook('Geesebook1_complete')
    # b = DatabaseBook('Pa1235_Hiwi_01')

    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.TEXT, True)], True, [e, f])

    #trainer_params.load = '/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/modules/ommr4all-server/internal_storage/default_models/french14/text_nautilus/best_accuracy.pth'

    params = DatasetParams(
        gt_required=True,
        height=64,
        text_image_color_type="color",
        pad=(5, 5, 5, 5),
        text_types=[BlockType.LYRICS]
    )
    train_params = AlgorithmTrainerSettings(
        params,
        train_pcgts + val_pcgts,
        val_pcgts + train_pcgts,
        params=AlgorithmTrainerParams(load="i/french14/text_guppy/text_guppy")
        )

    trainer = PytorchGuppyyTrainer(train_params)
    trainer.train(e)
