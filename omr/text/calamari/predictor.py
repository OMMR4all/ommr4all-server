from calamari_ocr.ocr.predictor import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams, Predictions
from typing import List, Tuple
from omr.dataset.datastructs import CalamariSequence
from database.file_formats import PcGts
from database import DatabaseBook
from omr.text.predictor import TextPredictorParameters, TextDatasetParams, \
    TextPredictor, PredictionType, TextDataset, \
    RegionLineMaskData, PredictionResult, Point
import numpy as np
from calamari_ocr.utils import glob_all


class CalamariPredictor(TextPredictor):
    def __init__(self, params: TextPredictorParameters):
        super().__init__(params)
        # self.predictor = MultiPredictor(glob_all([s + '/text_best*.ckpt.json' for s in params.checkpoints]))
        self.predictor = MultiPredictor(glob_all([s for s in params.checkpoints]))
        self.height = self.predictor.predictors[0].network_params.features
        voter_params = VoterParams()
        voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        self.voter = voter_from_proto(voter_params)

    def _predict(self, dataset: TextDataset) -> PredictionType:
        for marked_symbols, (r, sample) in zip(dataset.lines(), self.predictor.predict_dataset(dataset.to_text_line_calamari_dataset())):
            prediction = self.voter.vote_prediction_result(r)
            yield PredictionResult(self.extract_symbols(prediction, marked_symbols), marked_symbols)

    def extract_symbols(self, p, m: RegionLineMaskData) -> List[Tuple[str, Point]]:
        sentence = [(pos.chars[0].char,
                     self.dataset.line_and_mask_operations.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, 40), m.operation.params).x)
                    for pos in p.positions]
        return sentence


if __name__ == '__main__':
    import random
    from database.file_formats.pcgts.page.musicregion.musicline import Neume, SymbolType, GraphicalConnectionType
    import matplotlib.pyplot as plt
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState("Symbols", True), LockState("Layout", True)], True, [
        DatabaseBook('Graduel_Part_1'),
        DatabaseBook('Graduel_Part_2'),
        DatabaseBook('Graduel_Part_3'),
    ])
    params = TextDatasetParams(
        gt_required=True,
        height=80,
        cut_region=True,
        pad=(0, 10, 0, 20),
    )
    pred = CalamariPredictor(TextPredictorParameters(
        checkpoints=['/home/ls6/wick/Documents/Projects/calamari_models/fraktur_historical_ligs/*.ckpt.json'],
        # checkpoints=['calamari_models_out/pretraining/net_cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=1x2,cnn=80:3x3,pool=1x2,cnn=100:3x3,lstm=200,dropout=0.5/h136_folds0_center1_dewarp1/symbol_detection_0/best'],
        text_predictor_params=params,
    ))
    ps = list(pred.predict(val_pcgts[7:8]))
    orig = np.array(ps[0].line.operation.page_image)
    for p in ps:
        print("".join([t[0] for t in p.text]))

    plt.imshow(orig)
    plt.show()
