import os
from typing import List, Optional, Generator, Tuple

import torch
from PIL import Image
from guppyocr.predict_pxml import preprocess_image
from nautilus_ocr.decoder import DecoderOutput, DecoderType
from nautilus_ocr.predict import get_config, Network

from database.file_formats.pcgts import MusicSymbol, Point
from database.file_formats.performance.pageprogress import Locks
from ommr4all.settings import BASE_DIR
from omr.dataset.datastructs import CalamariSequence, RegionLineMaskData
from database.file_formats import PcGts
from database import DatabaseBook
import numpy as np

from omr.steps.symboldetection.dataset import SymbolDetectionDataset, SymbolDetectionDatasetTorch
from omr.steps.symboldetection.predictor import SymbolsPredictor, AlgorithmPredictorSettings, PredictionCallback, \
    SingleLinePredictionResult, PredictionResult
from omr.steps.symboldetection.sequence_to_sequence_guppy.api import GuppyOCR
from omr.steps.symboldetection.sequence_to_sequence_guppy.meta import Meta
from omr.steps.text.guppy.predictor import resize_with_pad, GreedyDecoder


class OMRPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.dict_corrector = None
        path = settings.model.local_file('model_best.pth')

        # print(path)
        # print(os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
        #                   'network_config', 'ocr_config.yaml'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        print(path)
        self.network = GuppyOCR.load_model(path, self.device)
        print(self.device)
        self.settings = settings
        self.dict_corrector = None
        self.database_hyphen_dictionary = None
        self.network.model.eval()

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[SingleLinePredictionResult, None, None]:

        # if self.database_hyphen_dictionary is None:
        #    db = DatabaseDictionary.load(book=book)
        #    self.database_hyphen_dictionary = db.to_hyphen_dict()

        dataset = SymbolDetectionDatasetTorch(pcgts_files, self.dataset_params)
        loaded_dataset = dataset.load()
        len_dataset = len(loaded_dataset)
        for i, y in enumerate(loaded_dataset):  # dataset_cal[0]:
            image = y.region
            if not len(image.shape) == 3:
                continue
                # raise InvalidInputImage("Len of image cutout shape must be 3")
            # print(self.network.mc.mconfig.Width)
            # print(self.network.mc.mconfig.Height)

            img, ratio = preprocess_image(image, self.network.mc.mconfig.Width, self.network.mc.mconfig.Height)
            # img2 = preprocess_image(image, self.network.mc.mconfig.Width, self.network.mc.mconfig.Height)

            img_t, sizes = resize_with_pad(image, (self.network.mc.mconfig.Width, self.network.mc.mconfig.Height),
                                           (255, 255, 255))
            img = img[None, :, :, :]
            with torch.no_grad():
                img = img.to(self.device)
                prediction, _, _, _ = self.network.model.forward(img, None)

                net_out = prediction[0].cpu().numpy()
                alphabet = [0] * len(self.network.mc.mconfig.id2char)
                for k, v in self.network.mc.mconfig.id2char.items():
                    alphabet[k] = v
                # from fast_ctc_decode import viterbi_search

                # seq, _ = viterbi_search(net_out, alphabet)
                g_decoder = GreedyDecoder(alphabet)
                sentence: DecoderOutput = g_decoder.decode(net_out, True, False, pad=sizes[1],debug_img=img_t)
                print(sentence.decoded_string)
            # print(self.dict_corrector)

            percentage = (i + 1) / len(loaded_dataset)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(loaded_dataset))

            delta = 1 - (sizes[1] / self.network.mc.mconfig.Width)
            # print(sentence)
            cc = self.extract_symbols(dataset, sentence, y, (self.network.mc.mconfig.Width) * (
                    image.shape[1] / self.network.mc.mconfig.Width) / (net_out.shape[0] * delta), pad=sizes[1])
            width = image.shape[1]
            if callback:
                percentage = (i + 1) / len_dataset

                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len_dataset)
            yield SingleLinePredictionResult(cc, y)







        #for marked_symbols, (r, sample) in zip(dataset.load(callback), self.predictor.predict_dataset(dataset.to_calamari_dataset())):
        #    prediction = self.voter.vote_prediction_result(r)
        #    yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, marked_symbols), marked_symbols)
    def extract_symbols(self, dataset, p, m: RegionLineMaskData, factor: float = 1, pad=0) -> List[
        MusicSymbol]:
        def i2p(p):
            return m.operation.page.image_to_page_scale(p, m.operation.scale_reference)

        dec_str = p.decoded_string
        positions = p.char_mapping
        sentence = []
        chars = []
        print("12323")
        for x in p.char_mapping.charLocationInfo:
            a = p.char_mapping.chars[x.char]
            sentence.append((a,
                             i2p(dataset.local_to_global_pos(Point(((x.char_start + x.char_end) / 2) * factor,
                                                                   40),
                                                             m.operation.params).x)))
        print(sentence)
        cc = CalamariSequence.to_symbols(dataset.params.calamari_codec, sentence,
                                         m.operation.music_line.staff_lines)
        return cc



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
    ps = list(pred.predict([p.page.location for p in val_pcgts[1:2]]))
    for p in ps:
        p: PredictionResult = p
        canvas = PcGtsCanvas(p.pcgts.page, PageScaleReference.NORMALIZED_X2)
        for sp in p.music_lines:
            canvas.draw(sp.symbols)

        canvas.show()
