import os
import json

from omr.dataset.datastructs import CalamariCodec
from omr.steps.symboldetection.trainer import SymbolDetectionTrainer

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from typing import Optional
from database import DatabaseBook
from omr.dataset import DatasetParams
from omr.steps.algorithm import TrainerCallback, AlgorithmTrainerParams, AlgorithmTrainerSettings
from omr.steps.symboldetection.sequencetosequence.meta import Meta

from database.file_formats.performance.pageprogress import Locks
from omr.steps.symboldetection.sequencetosequence.params import CalamariParams

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

        train_dataset = self.train_dataset.to_nautilus_dataset(train=True, callback=callback)
        val_dataset = self.validation_dataset.to_nautilus_dataset(train=False, callback=callback)



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



