import os
if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
from calamari_ocr.ocr.predictor import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams, Predictions
from typing import List, Tuple, Type, Optional, Generator

from database.file_formats.performance.pageprogress import Locks
from database.model import Model
from omr.dataset.datastructs import CalamariSequence, RegionLineMaskData
from database.file_formats import PcGts
from database import DatabaseBook
from omr.steps.algorithm import AlgorithmMeta, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.dataset import TextDataset
from omr.steps.text.predictor import \
    TextPredictor, \
    PredictionResult, Point, SingleLinePredictionResult
import numpy as np
from calamari_ocr.utils import glob_all
from database.model.definitions import MetaId


class CalamariPredictor(TextPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.calamari.meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        # self.predictor = MultiPredictor(glob_all([s + '/text_best*.ckpt.json' for s in params.checkpoints]))
        self.predictor = MultiPredictor(glob_all([settings.model.local_file('model.ckpt.json')]))
        self.height = self.predictor.predictors[0].network_params.features
        voter_params = VoterParams()
        voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        self.voter = voter_from_proto(voter_params)

    def _predict(self, dataset: TextDataset, callback: Optional[PredictionCallback]) -> Generator[SingleLinePredictionResult, None, None]:
        for marked_symbols, (r, sample) in zip(dataset.load(callback), self.predictor.predict_dataset(dataset.to_text_line_calamari_dataset())):
            prediction = self.voter.vote_prediction_result(r)
            yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, marked_symbols), marked_symbols)

    def extract_symbols(self, dataset: TextDataset, p, m: RegionLineMaskData) -> List[Tuple[str, Point]]:
        sentence = [(pos.chars[0].char,
                     dataset.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, 40), m.operation.params).x)
                    for pos in p.positions]
        return sentence


if __name__ == '__main__':
    from omr.steps.step import Step, AlgorithmTypes
    from ommr4all.settings import BASE_DIR
    import random
    import matplotlib.pyplot as plt
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    if False:
        train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.LAYOUT, True)], True, [
            # DatabaseBook('Graduel_Part_1'),
            # DatabaseBook('Graduel_Part_2'),
            # DatabaseBook('Graduel_Part_3'),
        ])
    book = DatabaseBook('New_York')
    meta = Step.meta(AlgorithmTypes.OCR_CALAMARI)
    # model = meta.newest_model_for_book(book)
    model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/pretrained_models/text_calamari/fraktur_historical', meta.type()))
    settings = AlgorithmPredictorSettings(
        model=model,
    )
    pred = meta.create_predictor(settings)
    ps: List[PredictionResult] = list(pred.predict(book.pages()))
    orig = np.array(ps[0].text_lines[0].line.line_image)
    f, ax = plt.subplots(len(ps), 1)
    for i, p in enumerate(ps):
        ax[i].imshow(p.text_lines[0].line.line_image)
        print("".join([t[0] for t in p.text_lines[0].text]))

    plt.show()
