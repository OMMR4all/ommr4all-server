import csv
from typing import Optional, Type, Dict, List, Tuple

import numpy as np
from nautilus_ocr.train import train

from database.file_formats.pcgts import BlockType
from database.file_formats.performance.pageprogress import Locks
from omr.dataset import DatasetParams, LyricsNormalization
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


def get_config(file_path, train_dir, val_dir, load, save_location, iter=300000):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt['train_data'] = train_dir
    opt['valid_data'] = val_dir
    opt['saved_model'] = load
    opt['model_save_location_dir'] = save_location
    opt['num_iter'] = iter
    if opt.lang_char == 'None':
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

        def create_nautilus_tempfiles(dir: str, dataset: Tuple[List[np.array], List[str]], type="train",
                                      subfolder: str = "train"):
            header = ["filename", "words"]
            path = os.path.join(dir, type, subfolder)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "labels.csv"), 'w', encoding='UTF8') as labels_csv:
                csv_writer = csv.writer(labels_csv)
                csv_writer.writerow(header)
                for ind, (image, gt) in enumerate(zip(dataset[0], dataset[1])):
                    Image.fromarray(image).save(os.path.join(path, str(ind) + ".png"))
                    with open(os.path.join(path, str(ind) + ".txt"), 'w', encoding='UTF8') as gt_txt:
                        gt_txt.write(gt)
                    csv_writer.writerow([os.path.join(str(ind) + ".png"), gt])

        train_dataset = self.train_dataset.to_text_line_nautilus_dataset(train=True, callback=callback)
        val_dataset = self.validation_dataset.to_text_line_nautilus_dataset(train=False, callback=callback)
        val_dataset = train_dataset if len(val_dataset[0]) == 0 else val_dataset
        with tempfile.TemporaryDirectory() as dirpath:
            loguru.logger.info(f"Creating temporary train directory at {dirpath}")
            create_nautilus_tempfiles(dirpath, train_dataset, type="train", subfolder="train")
            create_nautilus_tempfiles(dirpath, val_dataset, type="val", subfolder="val")
            opt = get_config(opt, os.path.join(dirpath, "train"), os.path.join(dirpath, "val"),
                             load='' if not self.params.model_to_load() else self.params.model_to_load().local_file('model.h5'),
                             save_location=self.settings.model.path)
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
    # b = DatabaseBook('Pa1235_Hiwi_01')

    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.LAYOUT, True)], True, [b])
    trainer_params = PytorchTrainer.default_params()
    trainer_params.l_rate = 1e-3
    # trainer_params.load = '/home/ls6/wick/Documents/Projects/calamari_models/fraktur_historical_ligs/0.ckpt.json'

    params = DatasetParams(
        gt_required=True,
        # height=64,
        # cut_region=True,
        # pad=[0, 10, 0, 20],
        # lyrics_normalization=LyricsNormalization.ONE_STRING,
    )
    train_params = AlgorithmTrainerSettings(
        params,
        train_pcgts,
        val_pcgts,
        params=trainer_params,
        calamari_params=CalamariParams(
            network="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
            n_folds=1,
        )
    )
    trainer = PytorchTrainer(train_params)
    trainer.train(b)
