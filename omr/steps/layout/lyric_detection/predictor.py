import os
from typing import List, Optional

import numpy as np
from PIL import Image

from database import DatabasePage
from database.file_formats import PcGts
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictionResultGenerator
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.layout.lyric_detection.dataset import LyricsLocationDataset
from omr.steps.layout.lyric_detection.meta import Meta
from omr.steps.layout.lyric_detection.helper_tools import Color_Palett_Reduction

# Detects red color in text images
class TextLocationDetector(AlgorithmPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def predict(self, pages: List[DatabasePage], callback: Optional = None) -> AlgorithmPredictionResultGenerator:
        pcgts_files = [p.pcgts() for p in pages]
        dataset = LyricsLocationDataset(pcgts_files, self.dataset_params).load()
        # dataset = pc_dataset.to_text_line_nautilus_dataset()
        cr = Color_Palett_Reduction()
        for i, y in enumerate(dataset):
            pass
            image = Image.fromarray(y.line_image)

            im = cr.reduced_quatnize(image)
            from matplotlib import pyplot as plt
            plt.imshow(np.array(im))
            plt.show()
            image.show()
            im.show()
    def unprocessed(cls, page: DatabasePage) -> bool:
        pass

if __name__ == "__main__":
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    from database import DatabaseBook

    b = DatabaseBook('mulhouse_mass_transcription')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()][7:8]
    pred = TextLocationDetector(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
