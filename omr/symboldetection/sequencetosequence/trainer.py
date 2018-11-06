from omr.symboldetection.thirdparty.calamari.calamari_ocr.ocr.trainer import Trainer
from omr.symboldetection.thirdparty.calamari.calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, network_params_from_definition_string
from omr.symboldetection.thirdparty.calamari.calamari_ocr.ocr.datasets import create_dataset, DataSetMode, DataSetType
from typing import List
from omr.datatypes import PcGts
from omr.dataset.pcgtsdataset import PcGtsDataset
import main.book as book


class OMRTrainer:
    def __init__(self, pcgts_files: List[PcGts]):
        self.pcgts_dataset = PcGtsDataset(pcgts_files, gt_required=True)

    def run(self):
        images = []
        gt = []
        for ml, img, s in self.pcgts_dataset.music_lines():
            images.append(img)
            gt.append(s)

        dataset = create_dataset(
            DataSetType.RAW, DataSetMode.TRAIN,
            images, gt
        )
        dataset.load_samples()
        params = CheckpointParams()

        params.max_iters = 100
        params.stats_size = 1000
        params.batch_size = 1
        params.checkpoint_frequency = 0
        params.output_dir = ''
        params.output_model_prefix = 'omr_'
        params.display = 0.5
        params.skip_invalid_gt = True
        params.processes = -1
        params.data_aug_retrain_on_original = True

        params.early_stopping_frequency = 1
        params.early_stopping_nbest = 5
        params.early_stopping_best_model_prefix = 'omr_best_'
        params.early_stopping_best_model_output_dir = ''

        params.model.data_preprocessor.type = DataPreprocessorParams.NOOP_NORMALIZER
        params.model.text_preprocessor.type = TextProcessorParams.NOOP_NORMALIZER
        params.model.text_postprocessor.type = TextProcessorParams.NOOP_NORMALIZER

        params.model.line_height = 80

        network_params_from_definition_string('cnn=10:3x3,pool=1x2,cnn=20:3x3,pool=1x2,lstm=20,dropout=0.5', params.model.network)
        trainer = Trainer(
            checkpoint_params=params,
            dataset=dataset
        )
        trainer.train()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_023')
    pcgts = PcGts.from_file(page.file('pcgts'))
    trainer = OMRTrainer([pcgts])
    trainer.run()



