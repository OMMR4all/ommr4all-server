from omr.symboldetection.thirdparty.calamari.calamari_ocr.ocr.trainer import Trainer
from omr.symboldetection.thirdparty.calamari.calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, network_params_from_definition_string
from omr.symboldetection.thirdparty.calamari.calamari_ocr.ocr.datasets import create_dataset, DataSetMode, DataSetType
from typing import List
from omr.datatypes import PcGts
from omr.dataset.pcgtsdataset import PcGtsDataset
import main.book as book


class OMRTrainer:
    def __init__(self, train_pcgts_files: List[PcGts], validation_pcgts_files: List[PcGts]):
        self.height = 80
        self.train_pcgts_dataset = PcGtsDataset(train_pcgts_files, gt_required=True, height=self.height)
        self.validation_pcgts_dataset = PcGtsDataset(validation_pcgts_files, gt_required=True, height=self.height)

    def run(self, model_for_book: book.Book):
        train_dataset = self.train_pcgts_dataset.to_calamari_dataset()
        val_dataset = self.validation_pcgts_dataset.to_calamari_dataset()

        params = CheckpointParams()

        params.max_iters = 1000000
        params.stats_size = 1000
        params.batch_size = 1
        params.checkpoint_frequency = 0
        params.output_dir = model_for_book.local_path('omr_models')
        params.output_model_prefix = 'omr'
        params.display = 100
        params.skip_invalid_gt = True
        params.processes = -1
        params.data_aug_retrain_on_original = True

        params.early_stopping_frequency = 1000
        params.early_stopping_nbest = 5
        params.early_stopping_best_model_prefix = 'omr_best'
        params.early_stopping_best_model_output_dir = model_for_book.local_path('omr_models')

        params.model.data_preprocessor.type = DataPreprocessorParams.NOOP_NORMALIZER
        params.model.text_preprocessor.type = TextProcessorParams.NOOP_NORMALIZER
        params.model.text_postprocessor.type = TextProcessorParams.NOOP_NORMALIZER

        params.model.line_height = self.height

        network_params_from_definition_string('cnn=32:3x3,pool=1x2,cnn=64:3x3,pool=1x2,lstm=100,dropout=0.5', params.model.network)
        trainer = Trainer(
            checkpoint_params=params,
            dataset=train_dataset,
            validation_dataset=val_dataset,
            n_augmentations=100,
            # weights='/home/ls6/wick/Documents/Projects/calamari/omr_models/mix.ckpt',
        )
        trainer.train()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    b = book.Book('Graduel')
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:3]]
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[3:4]]
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_023')
    # pcgts = PcGts.from_file(page.file('pcgts'))
    trainer = OMRTrainer(pcgts, val_pcgts)
    trainer.run(b)



