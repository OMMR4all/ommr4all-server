import abc
import logging
from typing import List

from database import DatabasePage, DatabaseBook
from database.file_formats import PcGts
from omr.dataset import DatasetParams
from omr.experimenter.experimenter import EvaluatorParams, SingleDataArgs, GlobalDataArgs
from omr.steps.symboldetection.evaluator import SymbolDetectionEvaluator, SymbolErrorTypeDetectionEvaluator
from omr.steps.text.experimenter import TextExperimenter
logger = logging.getLogger(__name__)


class PcGtsEvaluator():
    def __init__(self):
        pass
    @abc.abstractmethod
    def evaluate(self, pcgts_pred, pcgts_gt):
        pass

class SymbolPCGTSEvaluator(PcGtsEvaluator):
    pass

    def evaluate(self, pcgts_pred: List[PcGts], pcgts_gt: List[PcGts]):
        pred_symbols = [music_line.symbols for pcgts in pcgts_pred for music_line in pcgts.page.all_music_lines()]
        gt_symbols = [music_line.symbols for pcgts in pcgts_gt for music_line in pcgts.page.all_music_lines()]


        evaluator = SymbolDetectionEvaluator()
        res = evaluator.evaluate(gt_symbols, pred_symbols)
        print(res)
        evaluator2 = SymbolErrorTypeDetectionEvaluator()


class TextPCGTSEvaluator(PcGtsEvaluator):

    def evaluate(self, pcgts_pred: List[PcGts], pcgts_gt: List[PcGts]):
        pred_symbols = [music_line.sentence.text() for pcgts in pcgts_pred for music_line in pcgts.page.all_text_lines(only_lyric=True)]
        gt_symbols = [music_line.sentence.text() for pcgts in pcgts_gt for music_line in pcgts.page.all_text_lines(only_lyric=True)]
        args = SingleDataArgs(None, None, None, None, None, GlobalDataArgs(magic_prefix=None, model_dir=None, cross_folds=None,
                                                                           single_folds=None, skip_train=None, skip_predict=None, skip_eval=None, skip_cleanup=None,
                                                                           dataset_params=DatasetParams(), evaluation_params=EvaluatorParams(), predictor_params=None, output_book=None, output_debug_images=None,
                                                                           algorithm_type=None, trainer_params=None, page_segmentation_params=None, calamari_params=None, calamari_dictionary_from_gt=None))
        evaluator = TextExperimenter(args, logger)
        params = EvaluatorParams()
        res = evaluator.evaluate((gt_symbols, pred_symbols), params)
        print(res)

if __name__ == "__main__":
    book_pred = "mulhouse_mass_transcription"
    book_gt = "mulhouse2"
    pages = ["00008_Pa_904","00009_Pa_904","00010_Pa_904"]
    pages_gt = [DatabasePage(DatabaseBook(book_gt), x) for x in pages]
    pages_pred = [DatabasePage(DatabaseBook(book_pred), x) for x in pages]
    symbol_eval = SymbolPCGTSEvaluator()
    symbol_eval.evaluate([x.pcgts() for x in pages_pred], [x.pcgts() for x in pages_gt])
    text_eval = TextPCGTSEvaluator()
    text_eval.evaluate([x.pcgts() for x in pages_pred], [x.pcgts() for x in pages_gt])
    pass