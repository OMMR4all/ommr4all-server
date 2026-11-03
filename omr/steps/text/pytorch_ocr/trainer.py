import csv
import shutil
from typing import Optional, Type, Dict, List, Tuple

import numpy as np
from nautilus_ocr.train import train

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
from nautilus_ocr.utils import AttrDict


def get_config(file_path, train_dir, val_dir, load, save_location, iter=300000, chars=None):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt['train_data'] = train_dir
    opt['valid_data'] = val_dir
    opt['saved_model'] = load
    opt['model_save_location_dir'] = save_location
    opt['num_iter'] = iter
    if chars:
        opt.character = chars
    elif opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'],
                             keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    loguru.logger.info(f"Characters_in dataset: {opt.character}")
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


class CalamariTrainerCallback:
    def __init__(self, cb: TrainerCallback):
        self.cb = cb

    def display(self, train_cer, train_loss, train_dt, iter, steps_per_epoch, display_epochs,
                example_pred, example_gt):
        self.cb.next_iteration(iter, train_loss, 1 - train_cer)

    def early_stopping(self, eval_cer, n_total, n_best, iter):
        self.cb.next_best_model(iter, eval_cer, n_best - 1)


class PytorchTrainer(TextTrainerBase):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.pytorch_ocr.meta import Meta
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
        params.pad = (5, 5, 5, 5)
        params.text_types = [BlockType.LYRICS]

        # params.lyrics_normalization = params.lyrics_normalization.lyrics_normalization.WORDS

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        from ommr4all.settings import BASE_DIR

        opt = os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
                           'network_config', 'ocr_config.yaml')

        with tempfile.TemporaryDirectory() as dirpath:
            loguru.logger.info(f"Creating temporary train directory at {dirpath}")

            # the samples are written one by one as they are built, so the whole dataset
            # never has to fit into memory (the training library reads them from disk)
            n_train = write_line_samples(
                os.path.join(dirpath, "train", "train"),
                self.train_dataset.iter_text_line_nautilus_samples(train=True, callback=callback))
            n_val = write_line_samples(
                os.path.join(dirpath, "val", "val"),
                self.validation_dataset.iter_text_line_nautilus_samples(train=False, callback=callback))
            loguru.logger.info(f"Training samples: {n_train}, validation samples: {n_val}")
            if n_val == 0:
                # no validation data: validate on the training data, as before
                shutil.rmtree(os.path.join(dirpath, "val", "val"))
                shutil.copytree(os.path.join(dirpath, "train", "train"),
                                os.path.join(dirpath, "val", "val"))

            # opt = get_config(opt, os.path.join(dirpath, "train"), os.path.join(dirpath, "val"),
            #                 load='' if not self.params.model_to_load() else self.params.model_to_load().local_file(
            #                     'best_accuracy.pth'),
            #                 save_location=self.settings.model.path)
            opt = get_config(opt, os.path.join(dirpath, "train"), os.path.join(dirpath, "val"),
                             load=self.params.load,
                             save_location=self.settings.model.path,
                             chars=' \"#,.abcdefghiklmnopqrstuvxyzſω')
            while True:
                pass
            train(opt, amp=False)


if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(1)
    np.random.seed(1)

    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    import django

    django.setup()
    b = DatabaseBook('Graduel_Part_1_gt')
    c = DatabaseBook('Graduel_Part_2_gt')
    d = DatabaseBook('Graduel_Part_3_gt')
    e = DatabaseBook('Pa_14819_gt')
    f = DatabaseBook('Assisi')
    g = DatabaseBook('Cai_72')
    #c = DatabaseBook('mul_2_rsync_gt')

    # b = DatabaseBook('Pa1235_Hiwi_01')

    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.TEXT, True)], True, [b, c, d, e, f, g])
    print(len(train_pcgts))
    print(len(val_pcgts))
    print("2")
    for i in train_pcgts:
        print(i.page.location.local_path())
    trainer_params = PytorchTrainer.default_params()
    trainer_params.l_rate = 1e-4
    trainer_params.load = '/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/modules/ommr4all-server/internal_storage/default_models/french14/text_nautilus/best_accuracy.pth'

    params = DatasetParams(
        gt_required=True,
        height=64,
        text_image_color_type="color",
        pad=(5, 5, 5, 5),
        text_types=[BlockType.LYRICS]
    )
    train_params = AlgorithmTrainerSettings(
        params,
        train_pcgts ,
        val_pcgts ,
        params=trainer_params,
        calamari_params=CalamariParams(
            network="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
            n_folds=1,
        )
    )
    trainer = PytorchTrainer(train_params)
    trainer.train(c)
