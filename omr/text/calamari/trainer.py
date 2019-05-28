from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, network_params_from_definition_string
from calamari_ocr.ocr.trainer import Trainer
from calamari_ocr.ocr.cross_fold_trainer import CrossFoldTrainer
from calamari_ocr.ocr.augmentation import SimpleDataAugmenter
from typing import List, Optional
from database.file_formats import PcGts
from omr.text.dataset import TextDataset, TextDatasetParams
from omr.text.trainer import TextTrainerCallback, TextTrainerBase, TextTrainerParams, CalamariParams
from database import DatabaseBook
import os


class CalamariTrainer(TextTrainerBase):
    def __init__(self, params: TextTrainerParams):
        super().__init__(params)

    def run(self, model_for_book: Optional[DatabaseBook] = None, callback: Optional[TextTrainerCallback] = None):
        train_dataset = self.train_dataset.to_text_line_calamari_dataset(train=True)
        val_dataset = self.validation_dataset.to_text_line_calamari_dataset(train=True)
        output = self.params.output if self.params.output else model_for_book.local_path('text_models')

        params = CheckpointParams()

        params.max_iters = self.params.n_iter if self.params.n_iter > 0 else 1_000_000
        params.stats_size = 1000
        params.batch_size = 1
        params.checkpoint_frequency = 0
        params.output_dir = output
        params.output_model_prefix = 'text'
        params.display = self.params.display
        params.skip_invalid_gt = True
        params.processes = -1
        params.data_aug_retrain_on_original = True

        params.early_stopping_frequency = self.params.early_stopping_test_interval if self.params.early_stopping_test_interval > 0 else 1000
        params.early_stopping_nbest = self.params.early_stopping_max_keep if self.params.early_stopping_max_keep > 0 else 5
        params.early_stopping_best_model_prefix = 'text_best'
        params.early_stopping_best_model_output_dir = output

        params.model.data_preprocessor.type = DataPreprocessorParams.DEFAULT_NORMALIZER
        params.model.data_preprocessor.pad = 5
        params.model.data_preprocessor.line_height = self.params.dataset_params.height
        params.model.text_preprocessor.type = TextProcessorParams.NOOP_NORMALIZER
        params.model.text_postprocessor.type = TextProcessorParams.NOOP_NORMALIZER

        params.model.line_height = self.params.dataset_params.height

        network_str = self.params.calamari_params.network
        if self.params.l_rate > 0:
            network_str += ',learning_rate={}'.format(self.params.l_rate)

        if self.params.calamari_params.n_folds > 0:
            train_args = {
                "max_iters": params.max_iters,
                "stats_size": params.stats_size,
                "checkpoint_frequency": params.checkpoint_frequency,
                "pad": 0,
                "network": network_str,
                "early_stopping_frequency": params.early_stopping_frequency,
                "early_stopping_nbest": params.early_stopping_nbest,
                "line_height": params.model.line_height,
                "data_preprocessing": ["RANGE_NORMALIZER", "FINAL_PREPARATION"],
            }
            trainer = CrossFoldTrainer(
                self.params.calamari_params.n_folds, train_dataset,
                output, 'omr_best_{id}', train_args, progress_bars=True
            )
            temporary_dir = os.path.join(output, "temporary_dir")
            trainer.run(
                self.params.calamari_params.single_folds,
                temporary_dir=temporary_dir,
                spawn_subprocesses=False, max_parallel_models=1,    # Force to run in same scope as parent process
            )
        else:
            network_params_from_definition_string(network_str, params.model.network)
            trainer = Trainer(
                checkpoint_params=params,
                dataset=train_dataset,
                validation_dataset=val_dataset,
                n_augmentations=0,
                data_augmenter=SimpleDataAugmenter(),
                weights=self.params.load,
                preload_training=True,
                preload_validation=True,
            )
            trainer.train()


if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel')
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState("Layout", True)], True, [b])
    params = TextDatasetParams(
        gt_required=True,
        height=48,
        cut_region=True,
        pad=(0, 10, 0, 20),
    )
    train_params = TextTrainerParams(
        params,
        train_pcgts,
        val_pcgts,
        l_rate=1e-3,
        load='/home/ls6/wick/Documents/Projects/calamari_models/fraktur_historical_ligs/0.ckpt.json',
        calamari_params=CalamariParams(
            network="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5",
            n_folds=0,
        )
    )
    trainer = CalamariTrainer(train_params)
    trainer.run(b)



