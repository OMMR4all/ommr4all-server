import os
from dataclasses import dataclass
from itertools import groupby

import torch
from PIL import Image
from guppyocr.api import GuppyOCR, InvalidInputImage
from guppyocr.predict_pxml import preprocess_image
from nautilus_ocr.decoder import DecoderType, DecoderOutput

from database.database_dictionary import DatabaseDictionary
from omr.steps.text.guppy.meta import Meta

from nautilus_ocr.predict import Network, get_config
from omr.steps.text.correction_tools.dictionary_corrector.predictor import DictionaryCorrector
from ommr4all.settings import BASE_DIR
import cv2

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from typing import List, Tuple, Type, Optional, Generator
from omr.dataset.datastructs import RegionLineMaskData
from database import DatabaseBook
from omr.steps.algorithm import AlgorithmMeta, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.dataset import TextDataset
from omr.steps.text.predictor import \
    TextPredictor, \
    PredictionResult, Point, SingleLinePredictionResult
import numpy as np
from omr.steps.text.hyphenation.hyphenator import HyphenatorFromDictionary, Pyphenator, CombinedHyphenator, HyphenDicts

from typing import Generator


def resize_with_pad(image: np.ndarray,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int, int, int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(
        max(new / orig for new, orig in zip(new_shape, original_shape)))  # float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])

    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new / orig for new, orig in zip(new_shape, original_shape)))
        new_size = tuple([int(x * ratio) for x in original_shape])

    assert new_size[0] <= new_shape[0] and new_size[1] <= new_shape[1]
    image = cv2.resize(image, new_size)
    # delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
    # delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    # left, right = delta_w//2, delta_w-(delta_w//2)
    left, right = 0, delta_w
    # print(image.shape, top, bottom,left,right)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=padding_color)
    return image, (ratio, delta_w)


@dataclass
class CharLocationInfo:
    char_start: int
    char_end: int
    char: str


@dataclass
class ExtendedDecoderInfo:
    chars: str
    white_space_index: int
    blank_index: int
    charLocationInfo: List[CharLocationInfo]
    probability_map: np.array = None


@dataclass
class DecoderOutput:
    decoded_string: str
    char_mapping: ExtendedDecoderInfo = None


class GreedyDecoder():
    def __init__(self, chars):
        self.chars = chars

    def decode(self, mat, extendend_info=False):
        index_list = np.argmax(np.squeeze(mat), axis=1)
        blank_index = 0  # len(self.chars)
        best_chars_collapsed = [self.chars[k] for k, _ in groupby(index_list) if k != blank_index]
        pred_string = ''.join(best_chars_collapsed)
        extendend_info_stats = None
        if extendend_info:
            listextf = []

            start, end = 0, -1
            whitespace_char_index = self.chars.index(" ")
            prev = None
            for x, y in groupby(index_list):
                y_length = len(list(y))
                prev = x if prev is None else prev
                start = end + 1
                end = end + y_length
                listextf.append(CharLocationInfo(start, end, x))

            listextf = [x for x in listextf if x.char != blank_index]
            listextf = [x for x in listextf if x.char != whitespace_char_index]

            extendend_info_stats = ExtendedDecoderInfo(chars=self.chars,
                                                       white_space_index=whitespace_char_index,
                                                       blank_index=blank_index,
                                                       charLocationInfo=listextf,
                                                       probability_map=mat
                                                       )

        return DecoderOutput(decoded_string=pred_string, char_mapping=extendend_info_stats)


class GuppyPredictor(TextPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.guppy.meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.dict_corrector = None
        path = settings.model.local_file('model_best.pth')

        # print(path)
        # print(os.path.join(BASE_DIR, 'omr', 'steps', 'text', 'pytorch_ocr',
        #                   'network_config', 'ocr_config.yaml'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.network = GuppyOCR.load_model(path, self.device)
        print(self.device)
        self.settings = settings
        self.dict_corrector = None
        self.database_hyphen_dictionary = None
        self.network.model.eval()

    def _predict(self, dataset: TextDataset, callback: Optional[PredictionCallback] = None) -> Generator[
        SingleLinePredictionResult, None, None]:

        # print(callback)
        """
        hyphen = HyphenatorFromDictionary(
            dictionary=os.path.join(BASE_DIR, 'internal_storage', 'resources', 'hyphen_dictionary.txt'),
            normalization=dataset.params.lyrics_normalization,
        )
        """
        # dataset_cal = dataset.to_text_line_nautilus_dataset()
        book = dataset.files[0].dataset_page().book
        # if self.database_hyphen_dictionary is None:
        #    db = DatabaseDictionary.load(book=book)
        #    self.database_hyphen_dictionary = db.to_hyphen_dict()

        hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                                    right=1)

        if self.settings.params.useDictionaryCorrection:
            self.dict_corrector = DictionaryCorrector(hyphenator=hyphen)
            self.dict_corrector.load_dict(book=book)

        loaded_dataset = dataset.load()
        for i, y in enumerate(loaded_dataset):  # dataset_cal[0]:
            image = 255 - y.line_image
            if not len(image.shape) == 3:
                raise InvalidInputImage("Len of image cutout shape must be 3")

            img = preprocess_image(image, self.network.mc.mconfig.Width, self.network.mc.mconfig.Height)
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
                sentence: DecoderOutput = g_decoder.decode(net_out, True)

            if self.dict_corrector:
                if len(sentence.decoded_string) > 0:
                    hyphenated = self.dict_corrector.segmentate_correct_and_hyphenate_text(sentence.decoded_string)
                else:
                    hyphenated = sentence.decoded_string
            else:
                hyphenated = hyphen.apply_to_sentence(sentence.decoded_string)
            percentage = (i + 1) / len(loaded_dataset)
            if callback:
                callback.progress_updated(percentage, n_processed_pages=i + 1, n_pages=len(loaded_dataset))

            delta = 1 - (sizes[1] / self.network.mc.mconfig.Width)
            #print(sentence)
            text, chars = self.extract_symbols(dataset, sentence, y, (self.network.mc.mconfig.Width) * (
                        image.shape[1] / self.network.mc.mconfig.Width) / (net_out.shape[0] * delta), pad=sizes[1])
            width = image.shape[1]

            yield SingleLinePredictionResult(text=text,
                                             line=y, hyphenated=hyphenated, chars=chars)

    def extract_symbols(self, dataset: TextDataset, p, m: RegionLineMaskData, factor: float = 1, pad=0) -> List[
        Tuple[str, Point]]:
        def i2p(p):
            return m.operation.page.image_to_page_scale(p, m.operation.scale_reference)

        dec_str = p.decoded_string
        positions = p.char_mapping
        sentence = []
        chars = []
        for x in p.char_mapping.charLocationInfo:
            a = p.char_mapping.chars[x.char]
            sentence.append((a,
                             i2p(dataset.local_to_global_pos(Point(((x.char_start + x.char_end) / 2) * factor,
                                                                   m.operation.text_line.aabb.bottom()),
                                                             m.operation.params).x)))
            chars.append((a,
                          [i2p(dataset.local_to_global_pos(Point(x.char_start * factor,
                                                                 m.operation.text_line.aabb.bottom()),
                                                           m.operation.params)),
                           i2p(dataset.local_to_global_pos(Point(x.char_end * factor,
                                                                 m.operation.text_line.aabb.bottom()),
                                                           m.operation.params))

                           ]
                          )
                         )

        return sentence, chars


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
    book = DatabaseBook('mul_2_rsync_base_symbol_finetune_w_pp')
    meta = Step.meta(AlgorithmTypes.OCR_GUPPY)
    #load = "i/french14/text_guppy/text_guppy")
    from database.model import Model

    model = Model.from_id_str("e/mul_2_rsync_gt/text_guppy/2023-07-05T16:32:45")
    # model = meta.newest_model_for_book(book)
    # model = Model(
    #    MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical',
    ##                            meta.type()))
    pred = GuppyPredictor(AlgorithmPredictorSettings(model=model))
    ps: List[PredictionResult] = list(pred.predict(book.pages()[92:93]))
    for i, p in enumerate(ps):
        canvas = PcGtsCanvas(p.pcgts.page, p.text_lines[0].line.operation.scale_reference)
        for j, s in enumerate(p.text_lines):
            canvas.draw(s)

        canvas.show()
