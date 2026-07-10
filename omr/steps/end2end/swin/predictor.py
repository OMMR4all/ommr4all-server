import logging
from typing import Type

import torch
from PIL import Image

from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.network import SwinTransformerOMR
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictorSettings

from ..predictor import End2EndPredictor
from .torch_dataset import preprocess_pil_image, DEFAULT_HEIGHT

logger = logging.getLogger(__name__)

MAX_DECODE_LEN = 1500


class SwinPredictor(End2EndPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from .meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        if settings.model is None:
            raise FileNotFoundError("No end2end model available. Train a model for this book first.")
        super().__init__(settings)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = SmartTokenizer(self.settings.model.local_file('codec.txt'))
        self.model = SwinTransformerOMR(vocab_size=len(self.tokenizer))
        state = torch.load(self.settings.model.local_file('model_best.pth'), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.height = self.dataset_params.height if self.dataset_params.height > 0 else DEFAULT_HEIGHT

    def _predict_crop(self, image: Image.Image) -> str:
        img = preprocess_pil_image(image, self.height).unsqueeze(0).to(self.device)

        curr_seq = torch.tensor([[self.tokenizer.SOS]], device=self.device)
        predicted_ids = []
        with torch.no_grad():
            for _ in range(MAX_DECODE_LEN):
                logits = self.model(img, curr_seq)
                next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if next_token_id == self.tokenizer.EOS:
                    break
                predicted_ids.append(next_token_id)
                curr_seq = torch.cat(
                    [curr_seq, torch.tensor([[next_token_id]], device=self.device)], dim=1)

        return self.tokenizer.decode(predicted_ids)


if __name__ == '__main__':
    import os

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    import django

    django.setup()

    from database import DatabaseBook
    from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
    from .meta import Meta

    book = DatabaseBook('Graduel_Part_1_gt')
    model = Meta.newest_model_for_book(book)
    settings = AlgorithmPredictorSettings(model=model, params=AlgorithmPredictorParams())
    predictor = SwinPredictor(settings)
    for result in predictor.predict(book.pages()[:1]):
        for b in result.blocks:
            print(b.music_line.id, len(b.symbols), b.sentence.text() if b.sentence else None)
