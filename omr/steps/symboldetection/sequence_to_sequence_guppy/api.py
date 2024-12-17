from typing import Union, Tuple

import torch

from abc import ABC, abstractmethod
from os import PathLike
import numpy.typing

from guppyocr.dataset.dataset import resize_with_pad
from guppyocr.decoder import GreedyDecoder, DecoderOutput
from guppyocr.train_calamares import decode_ctc
from guppyocr.predict_pxml import preprocess_image

from omr.steps.symboldetection.sequence_to_sequence_guppy.arch import ModelConfiguration


class OCREngineBase(ABC):
    @abstractmethod
    def ocr(self, cutout: numpy.typing.ArrayLike) -> str:
        pass


class InvalidInputImage(Exception):
    pass


class GuppyOCR(OCREngineBase):
    def __init__(self, mc: ModelConfiguration, model: torch.nn.Module, device: torch.device):
        self.mc = mc
        self.model = model
        self.device = device
        self.model.eval()

    @staticmethod
    def load_model(file: Union[str, PathLike], device: torch.device = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        mc, model = ModelConfiguration.load_model(file, device)
        return GuppyOCR(mc, model, device)

    def ocr(self, cutout: numpy.typing.ArrayLike) -> str:
        if not len(cutout.shape) == 3:
            raise InvalidInputImage("Len of image cutout shape must be 3")

        img, _ = preprocess_image(cutout, self.mc.mconfig.Width, self.mc.mconfig.Height)
        img = img[None, :, :, :]
        with torch.no_grad():
            img = img.to(self.device)
            prediction, _, _, _ = self.model.forward(img, None)
            decoded_tgt = decode_ctc(prediction[0], self.mc.mconfig.id2char)
        return decoded_tgt

    def ocr_with_char_position(self, cutout: numpy.typing.ArrayLike) -> Tuple[DecoderOutput, float]:
        if not len(cutout.shape) == 3:
            raise InvalidInputImage("Len of image cutout shape must be 3")

        img, (img_scale_ratio, img_scale_delta_w) = preprocess_image(cutout, self.mc.mconfig.Width,
                                                                     self.mc.mconfig.Height)
        # img2 = preprocess_image(image, self.network.mc.mconfig.Width, self.network.mc.mconfig.Height)

        img = img[None, :, :, :]
        with torch.no_grad():
            img = img.to(self.device)
            prediction, _, _, _ = self.model.forward(img, None)

            net_out = prediction[0].cpu().numpy()
            alphabet = [0] * len(self.mc.mconfig.id2char)
            for k, v in self.mc.mconfig.id2char.items():
                alphabet[k] = v

            sentence: DecoderOutput = GreedyDecoder(alphabet).decode(net_out, extended_info=True)
            delta = 1 - (img_scale_delta_w / self.mc.mconfig.Width)

            local_to_global_pos_factor = (self.mc.mconfig.Width) * (
                    cutout.shape[1] / self.mc.mconfig.Width) / (net_out.shape[0] * delta)

            return sentence, local_to_global_pos_factor

