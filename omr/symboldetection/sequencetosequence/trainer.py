from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, network_params_from_definition_string
from calamari_ocr.ocr.trainer import Trainer
from calamari_ocr.ocr.augmentation import SimpleDataAugmenter
from typing import List, Optional
from database.file_formats import PcGts
from omr.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetParams
from omr.symboldetection.trainer import SymbolDetectionTrainerCallback, SymbolDetectionTrainerBase, SymbolDetectionTrainerParams
from database import DatabaseBook


class OMRTrainer(SymbolDetectionTrainerBase):
    def __init__(self, params: SymbolDetectionTrainerParams):
        super().__init__(params)

    def run(self, model_for_book: Optional[DatabaseBook] = None, callback: Optional[SymbolDetectionTrainerCallback] = None):
        train_dataset = self.params.train_data.to_music_line_calamari_dataset()
        val_dataset = self.params.validation_data.to_music_line_calamari_dataset()
        output = self.params.output if self.params.output else model_for_book.local_path('omr_models')

        params = CheckpointParams()

        params.max_iters = self.params.n_iter if self.params.n_iter > 0 else 1_000_000
        params.stats_size = 1000
        params.batch_size = 1
        params.checkpoint_frequency = 0
        params.output_dir = output
        params.output_model_prefix = 'omr'
        params.display = self.params.display
        params.skip_invalid_gt = True
        params.processes = -1
        params.data_aug_retrain_on_original = True

        params.early_stopping_frequency = self.params.early_stopping_test_interval if self.params.early_stopping_test_interval > 0 else 1000
        params.early_stopping_nbest = self.params.early_stopping_max_keep if self.params.early_stopping_max_keep > 0 else 5
        params.early_stopping_best_model_prefix = 'omr_best'
        params.early_stopping_best_model_output_dir = output

        params.model.data_preprocessor.type = DataPreprocessorParams.FINAL_PREPARATION
        params.model.text_preprocessor.type = TextProcessorParams.NOOP_NORMALIZER
        params.model.text_postprocessor.type = TextProcessorParams.NOOP_NORMALIZER

        params.model.line_height = self.params.train_data.params.height

        network_str = self.params.calamari_params.network
        if self.params.l_rate > 0:
            network_str += ',learning_rate={}'.format(self.params.l_rate)

        network_params_from_definition_string(network_str, params.model.network)
        trainer = Trainer(
            checkpoint_params=params,
            dataset=train_dataset,
            validation_dataset=val_dataset,
            n_augmentations=0,
            data_augmenter=SimpleDataAugmenter(),
            weights=self.params.load
        )
        trainer.train()


if __name__ == '__main__':
    b = DatabaseBook('Graduel')
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState("Symbols", True), LockState("Layout", True)], True, [b])
    params = SymbolDetectionDatasetParams(
        gt_required=True,
        height=80,
        dewarp=True,
        cut_region=True,
        pad=(0, 40, 0, 80),
        center=True,
        staff_lines_only=True,
    )
    train_params = SymbolDetectionTrainerParams(
        SymbolDetectionDataset(train_pcgts, params),
        SymbolDetectionDataset(val_pcgts, params),
    )
    trainer = OMRTrainer(train_params)
    trainer.run(b)



