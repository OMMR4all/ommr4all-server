from calamari_ocr.ocr.predictor import Predictor, MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams, Predictions
from typing import List

from database.file_formats.performance.pageprogress import Locks
from omr.dataset.datastructs import CalamariSequence
from database.file_formats import PcGts
from database import DatabaseBook
from omr.symboldetection.predictor import SymbolDetectionPredictor, SymbolDetectionPredictorParameters, \
    SymbolDetectionDataset, PredictionType, SymbolDetectionDatasetParams, \
    RegionLineMaskData, Symbol, PredictionResult, Point
import numpy as np
from calamari_ocr.utils import glob_all


class OMRPredictor(SymbolDetectionPredictor):
    def __init__(self, params: SymbolDetectionPredictorParameters):
        super().__init__(params)
        self.predictor = MultiPredictor(glob_all([s + '/omr_best*.ckpt.json' for s in params.checkpoints]))
        self.height = self.predictor.predictors[0].network_params.features
        voter_params = VoterParams()
        voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        self.voter = voter_from_proto(voter_params)

    def _predict(self, dataset: SymbolDetectionDataset) -> PredictionType:
        for marked_symbols, (r, sample) in zip(dataset.marked_symbols(), self.predictor.predict_dataset(dataset.to_music_line_calamari_dataset())):
            prediction = self.voter.vote_prediction_result(r)
            yield PredictionResult(self.extract_symbols(prediction, marked_symbols), marked_symbols)

    def extract_symbols(self, p, m: RegionLineMaskData) -> List[Symbol]:
        sentence = [(pos.chars[0].char,
                     self.dataset.line_and_mask_operations.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, 40), m.operation.params).x)
                    for pos in p.positions]
        return CalamariSequence.to_symbols(sentence, m.operation.music_line.staff_lines)


if __name__ == '__main__':
    import random
    from database.file_formats.pcgts.page.musicregion.musicline import Neume, SymbolType, GraphicalConnectionType
    import matplotlib.pyplot as plt
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True)], True, [
        DatabaseBook('Graduel_Part_1'),
        DatabaseBook('Graduel_Part_2'),
        DatabaseBook('Graduel_Part_3'),
    ])
    params = SymbolDetectionDatasetParams(
        gt_required=True,
        height=136,
        dewarp=True,
        cut_region=False,
        pad=(0, 10, 0, 20),
        center=True,
        staff_lines_only=True,
    )
    pred = OMRPredictor(SymbolDetectionPredictorParameters(
        #checkpoints=['models_out/all_s2s/symbol_detection_0/best'],
        #checkpoints=['storage/Graduel_Fully_Annotated/omr_models/'],
        checkpoints=['calamari_models_out/pretraining/net_cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=1x2,cnn=80:3x3,pool=1x2,cnn=100:3x3,lstm=200,dropout=0.5/h136_folds0_center1_dewarp1/symbol_detection_0/best'],
        symbol_detection_params=params,
    ))
    ps = list(pred.predict(val_pcgts[7:8]))
    orig = np.array(ps[0].line.operation.page_image)
    for p in ps:
        for s in p.symbols:
            if s.symbol_type == SymbolType.NEUME:
                n: Neume = s
                for i, nc in enumerate(n.notes):
                    c = nc.coord.round().astype(int)
                    t, l = c.y, c.x
                    orig[t - 2:t + 2, l-2:l+2] = 255 * (i != 0) if nc.graphical_connection == GraphicalConnectionType.LOOPED else 250
            else:
                c = s.coord.round().astype(int)
                t, l = c.y, c.x
                orig[t - 2:t + 2, l-2:l+2] = 0

    plt.imshow(orig)
    plt.show()
