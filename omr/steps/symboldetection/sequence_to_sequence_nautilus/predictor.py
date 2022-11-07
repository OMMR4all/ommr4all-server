import os
from typing import List, Optional, Generator

from PIL import Image
from nautilus_ocr.decoder import DecoderOutput, DecoderType
from nautilus_ocr.predict import get_config, Network

from database.file_formats.pcgts import MusicSymbol, Point
from database.file_formats.performance.pageprogress import Locks
from ommr4all.settings import BASE_DIR
from omr.dataset.datastructs import CalamariSequence, RegionLineMaskData
from database.file_formats import PcGts
from database import DatabaseBook
import numpy as np

from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.steps.symboldetection.predictor import SymbolsPredictor, AlgorithmPredictorSettings, PredictionCallback, \
    SingleLinePredictionResult, PredictionResult
from omr.steps.symboldetection.sequence_to_sequence_nautilus.meta import Meta


class OMRPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.dict_corrector = None
        self.chars = '!#$%&\')*+-.0123456789:;=?@[\]^_`abcdefglmnpqrstuvwx{|'
        path = settings.model.local_file('best_accuracy.pth')

        opt = get_config(os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
                                      'network_config', 'ocr_config.yaml'), self.chars)
        self.network = Network(opt, path, self.chars, corpus='')

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDataset(pcgts_files, self.dataset_params)
        for y in dataset.load():  # dataset_cal[0]:
            image = Image.fromarray(y.region).convert('L')
            #from matplotlib import pyplot as plt
            #plt.imshow(image)
            #plt.show()

            sentence: DecoderOutput = self.network.predict_single_image(image, decoder_type=DecoderType(
                DecoderType.greedy_decoder))

            hidden_size = sentence.char_mapping.probability_map.shape[0]
            width = image.size[0]
            #print(sentence.decoded_string)
            yield SingleLinePredictionResult(self.extract_symbols(dataset, sentence, y, width / hidden_size), y)
        #for marked_symbols, (r, sample) in zip(dataset.load(callback), self.predictor.predict_dataset(dataset.to_calamari_dataset())):
        #    prediction = self.voter.vote_prediction_result(r)
        #    yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, marked_symbols), marked_symbols)
    def extract_symbols(self, dataset, p, m: RegionLineMaskData, factor: float = 1) -> List[MusicSymbol]:
        def i2p(p):
            return m.operation.page.image_to_page_scale(p, m.operation.scale_reference)

        dec_str = p.decoded_string
        positions = p.char_mapping
        sentence = []
        for x in p.char_mapping.charLocationInfo:
            a = p.char_mapping.chars[x.char]
            sentence.append((a,
                             i2p(dataset.local_to_global_pos(Point((x.char_start + x.char_end) / 2 * factor,
                                                                   40),
                                                             m.operation.params).x)))
        cc = CalamariSequence.to_symbols(dataset.params.calamari_codec, sentence, m.operation.music_line.staff_lines)
        return cc
    #def extract_symbols(self, dataset, p, m: RegionLineMaskData) -> List[MusicSymbol]:
    #    sentence = [(pos.chars[0].char,
    #                 m.operation.page.image_to_page_scale(
    #                     dataset.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, 40), m.operation.params).x,
    #                     m.operation.scale_reference
    #                 ))
    #                for pos in p.positions]
    #    return CalamariSequence.to_symbols(dataset.params.calamari_codec, sentence, m.operation.music_line.staff_lines)


if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    import random
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    from shared.pcgtscanvas import PcGtsCanvas, PageScaleReference
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel_Part_1')
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True)], True, [
        DatabaseBook('Graduel_Part_1'),
        DatabaseBook('Graduel_Part_2'),
        #DatabaseBook('Graduel_Part_3'),
    ])

    pred = OMRPredictor(AlgorithmPredictorSettings(
        model=Meta.best_model_for_book(b),
    ))
    ps = list(pred.predict([p.page.location for p in val_pcgts[2:3]]))
    for p in ps:
        p: PredictionResult = p
        canvas = PcGtsCanvas(p.pcgts.page, PageScaleReference.NORMALIZED_X2)
        for sp in p.music_lines:
            canvas.draw(sp.symbols)

        canvas.show()
