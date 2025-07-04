import csv
import os
import json
import tempfile

import loguru
import torch
from PIL import Image
import numpy as np
from omr.dataset.datastructs import CalamariCodec
from omr.steps.symboldetection.sequence_to_sequence_guppy.arch import train_model
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

        #train_dataset = self.train_dataset.to_nautilus_dataset(train=True, callback=callback)
        #val_dataset = self.validation_dataset.to_nautilus_dataset(train=False, callback=callback)
        from ommr4all.settings import BASE_DIR
        print(1)

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

        train_dataset = self.train_dataset.to_guppy_symbol_line__dataset(train=True, callback=callback, only_with_gt=True)
        val_dataset = self.validation_dataset.to_guppy_symbol_line__dataset(train=True, callback=callback, only_with_gt=True)

        #print("Lneghtd")
        #print(len(val_dataset[0]))
        val_dataset = train_dataset if len(val_dataset[0]) == 0 else val_dataset
        with tempfile.TemporaryDirectory() as dirpath:
            loguru.logger.info(f"Creating temporary train directory at {dirpath}")

            create_tempfiles(dirpath, train_dataset, type="", subfolder="train")
            create_tempfiles(dirpath, val_dataset, type="", subfolder="test")
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



