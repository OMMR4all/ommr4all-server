from calamari_ocr.ocr.predictor import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams
from typing import List, Optional, Generator

from database.file_formats.pcgts import MusicSymbol, Point
from database.file_formats.performance.pageprogress import Locks
from omr.dataset.datastructs import CalamariSequence, RegionLineMaskData
from database.file_formats import PcGts
from database import DatabaseBook
import numpy as np
from calamari_ocr.utils import glob_all

from omr.steps.symboldetection.dataset import SymbolDetectionDataset
from omr.steps.symboldetection.predictor import SymbolsPredictor, AlgorithmPredictorSettings, PredictionCallback, \
    SingleLinePredictionResult, PredictionResult
from omr.steps.symboldetection.sequencetosequence.meta import Meta


class OMRPredictor(SymbolsPredictor):
    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.predictor = MultiPredictor(glob_all([s + '/omr_best*.ckpt.json' for s in [settings.model.path]]))
        self.height = self.predictor.predictors[0].network_params.features
        voter_params = VoterParams()
        voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        self.voter = voter_from_proto(voter_params)

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> Generator[SingleLinePredictionResult, None, None]:
        dataset = SymbolDetectionDataset(pcgts_files, self.dataset_params)
        for marked_symbols, (r, sample) in zip(dataset.load(callback), self.predictor.predict_dataset(dataset.to_calamari_dataset())):
            prediction = self.voter.vote_prediction_result(r)
            yield SingleLinePredictionResult(self.extract_symbols(dataset, prediction, marked_symbols), marked_symbols)

    def extract_symbols(self, dataset, p, m: RegionLineMaskData) -> List[MusicSymbol]:
        sentence = [(pos.chars[0].char,
                     m.operation.page.image_to_page_scale(
                         dataset.local_to_global_pos(Point((pos.global_start + pos.global_end) / 2, 40), m.operation.params).x,
                         m.operation.scale_reference
                     ))
                    for pos in p.positions]
        return CalamariSequence.to_symbols(dataset.params.calamari_codec, sentence, m.operation.music_line.staff_lines)


if __name__ == '__main__':
    import random
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    from shared.pcgtscanvas import PcGtsCanvas, PageScaleReference
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel_Fully_Annotated')
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True)], True, [
        DatabaseBook('Graduel_Part_1'),
        DatabaseBook('Graduel_Part_2'),
        DatabaseBook('Graduel_Part_3'),
    ])
    pred = OMRPredictor(AlgorithmPredictorSettings(
        model=Meta.best_model_for_book(b),
    ))
    ps = list(pred.predict([p.page.location for p in val_pcgts[7:8]]))
    for p in ps:
        p: PredictionResult = p
        canvas = PcGtsCanvas(p.pcgts.page, PageScaleReference.NORMALIZED_X2)
        for sp in p.music_lines:
            canvas.draw(sp.symbols)

        canvas.show()
