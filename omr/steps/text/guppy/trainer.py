import csv
from typing import Optional, Type, Dict, List, Tuple

import numpy as np
import torch
from guppyocr.train_calamares import train_model
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
        params.pad = (5, 5, 5, 5)
        params.text_types = [BlockType.LYRICS]

        # params.lyrics_normalization = params.lyrics_normalization.lyrics_normalization.WORDS

    def __init__(self, settings: AlgorithmTrainerSettings):
        super().__init__(settings)

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        from ommr4all.settings import BASE_DIR


        def create_tempfiles(dir: str, dataset: Tuple[List[np.array], List[str]], type="train",
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

            # exit()
            create_tempfiles(dirpath, train_dataset, type="", subfolder="train")
            create_tempfiles(dirpath, val_dataset, type="", subfolder="test")
            from guppyocr.train_calamares import TrainingOpts
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print( self.params.model_to_load().local_file('model_best.pth'))
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
                epoch=250,
                grad_clip=False
            )
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
    h = DatabaseBook('mul_2_rsync_gt')

    # b = DatabaseBook('Pa1235_Hiwi_01')

    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.TEXT, True)], True, [h])
    print(len(train_pcgts))
    print(len(val_pcgts))
    print("2")
    for i in train_pcgts:
        print(i.page.location.local_path())
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
    trainer.train(h)
