import os

import scipy
from dataclasses import field, dataclass

import torch
from PIL import Image
from nautilus_ocr.decoder import DecoderType, DecoderOutput
from symspellpy import SymSpell, Verbosity

from omr.steps.text.pytorch_ocr.meta import Meta

# from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalization, LyricsNormalizationParams
from ctc_decoder import LanguageModel, beam_search, best_path
from nautilus_ocr.predict import Network, get_config
from tfaip.data.databaseparams import DataPipelineParams

from omr.steps.text.calamari.calamari_interface import RawData
from omr.steps.text.correction_tools.dictionary_corrector.predictor import DictionaryCorrector

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from typing import List, Tuple, Type, Optional, Generator
from ommr4all.settings import BASE_DIR
from copy import deepcopy

from database.model import Model
from omr.dataset.datastructs import RegionLineMaskData
from database import DatabaseBook
from omr.steps.algorithm import AlgorithmMeta, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.dataset import TextDataset
from omr.steps.text.predictor import \
    TextPredictor, \
    PredictionResult, Point, SingleLinePredictionResult
import numpy as np
from database.model.definitions import MetaId
from omr.steps.text.hyphenation.hyphenator import HyphenatorFromDictionary, Pyphenator

from typing import Generator


class PytorchPredictor(TextPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.calamari.meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.dict_corrector = None
        self.chars = ' "#,.abcdefghiklmnopqrstuvxyzſω'
        path = settings.model.local_file('best_accuracy.pth')

        opt = get_config(os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
                                      'network_config', 'ocr_config.yaml'), self.chars)
        print(path)
        print(os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
                           'network_config', 'ocr_config.yaml'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.network = Network(opt, path, self.chars, corpus='')
        if settings.params.useDictionaryCorrection:
            self.dict_corrector = DictionaryCorrector()

    def _predict(self, dataset: TextDataset, callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:
        hyphen = Pyphenator()
        """
        hyphen = HyphenatorFromDictionary(
            dictionary=os.path.join(BASE_DIR, 'internal_storage', 'resources', 'hyphen_dictionary.txt'),
            normalization=dataset.params.lyrics_normalization,
        )
        """
        dataset_cal = dataset.to_text_line_nautilus_dataset()
        book = dataset.files[0].dataset_page().book
        if self.dict_corrector:
            self.dict_corrector.load_dict(book=book)
        path = os.path.join(BASE_DIR, 'tools', 'sentence_dictionary.json')
        exists = os.path.exists(path)
        ###with open(path, 'r') as file:
        #    text = file.read().replace('\n', '')
        chars = " #,.abcdefghiklmnopqrstuvxyzſω"
        #sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        #loaded = sym_spell.load_dictionary(path, term_index=0, count_index=1, separator="$")

        # lm = LanguageModel(text, chars)
        # data_params = dataset_cal
        # dictionary = os.path.join(BASE_DIR, 'tools', 'sentence_dictionary.json'),

        for y in dataset.load():  # dataset_cal[0]:
            image = Image.fromarray(255 - y.line_image).convert('L')

            sentence: DecoderOutput = self.network.predict_single_image(image, decoder_type=DecoderType(
                DecoderType.greedy_decoder))

            hidden_size = sentence.char_mapping.probability_map.shape[0]
            width = image.size[0]
            if self.dict_corrector:
                if len(sentence.decoded_string) > 0:
                    hyphenated = self.dict_corrector.segmentate_correct_and_hyphenate_text(sentence.decoded_string)
                else:
                    hyphenated = sentence.decoded_string
            else:
                hyphenated = hyphen.apply_to_sentence(sentence.decoded_string)
            yield SingleLinePredictionResult(self.extract_symbols(dataset, sentence, y, width / hidden_size),
                                             y, hyphenated=hyphenated)

    def extract_symbols(self, dataset: TextDataset, p, m: RegionLineMaskData, factor: float = 1) -> List[
        Tuple[str, Point]]:
        def i2p(p):
            return m.operation.page.image_to_page_scale(p, m.operation.scale_reference)

        dec_str = p.decoded_string
        positions = p.char_mapping
        sentence = []
        for x in p.char_mapping.charLocationInfo:
            a = p.char_mapping.chars[x.char]
            sentence.append((a,
                             i2p(dataset.local_to_global_pos(Point((x.char_start + x.char_end) / 2 * factor,
                                                                   m.operation.text_line.aabb.bottom()),
                                                             m.operation.params).x)))

        return sentence


if __name__ == '__main__':
    from omr.steps.step import Step, AlgorithmTypes
    from ommr4all.settings import BASE_DIR
    import random
    import cv2
    import matplotlib.pyplot as plt
    from shared.pcgtscanvas import PcGtsCanvas
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True),
                                                               LockState(Locks.LAYOUT, True)], True, [
                                                             # DatabaseBook('Graduel_Part_1'),
                                                             # DatabaseBook('Graduel_Part_2'),
                                                             # DatabaseBook('Graduel_Part_3'),
                                                         ])
    book = DatabaseBook('Graduel_Part_1')
    meta = Step.meta(AlgorithmTypes.OCR_NAUTILUS)
    # model = meta.newest_model_for_book(book)
    # model = Model(
    #    MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical',
    ##                            meta.type()))
    pred = PytorchPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(book)))
    ps: List[PredictionResult] = list(pred.predict(book.pages()[0:1]))
    for i, p in enumerate(ps):
        canvas = PcGtsCanvas(p.pcgts.page, p.text_lines[0].line.operation.scale_reference)
        for j, s in enumerate(p.text_lines):
            canvas.draw(s)

        canvas.show()
