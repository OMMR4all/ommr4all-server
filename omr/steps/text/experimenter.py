from typing import List, Tuple

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts.page import Annotations, Sentence
from omr.dataset import LyricsNormalization
from omr.dataset.dataset import LyricsNormalizationProcessor
from omr.experimenter.experimenter import Experimenter
from .predictor import PredictionResult
from prettytable import PrettyTable
import numpy as np
from edit_distance import edit_distance

from .text_synchronizer import synchronize


class EvaluatorParams:
    progress_bar: bool = True
    skip_empty_gt: bool = False
    non_existing_pred_as_empty: bool = True


class Evaluator:
    def __init__(self, params: EvaluatorParams):
        """Class to evaluation the CER and errors of two dataset"""
        self.params = params

    @staticmethod
    def evaluate_single_args(args):
        return Evaluator.evaluate_single(**args)

    @staticmethod
    def evaluate_single(_sentinel=None, gt="", pred="", skip_empty_gt=False):
        """Evaluate a single pair of data

        Parameters
        ----------
        _sentinel : None
            Sentinel to force to specify gt and pred manually
        gt : str
            ground truth
        pred : str
            prediction
        skip_empty_gt : bool
            skip gt text lines that are empty

        Returns
        -------
        int
            length of ground truth
        int
            number of errors
        int
            number of synchronisation errors
        dict
            confusions dictionary
        tuple(str, str)
            ground_truth, prediction (same as input)

        """
        confusion = {}
        total_sync_errs = 0

        if len(gt) == 0 and skip_empty_gt:
            return 0, 0, 0, confusion, (gt, pred)

        errs, trues = edit_distance(gt, pred)
        synclist = synchronize([gt, pred])
        for sync in synclist:
            gt_str, pred_str = sync.get_text()
            if gt_str != pred_str:
                key = (gt_str, pred_str)
                total_sync_errs += max(len(gt_str), len(pred_str))
                if key not in confusion:
                    confusion[key] = 1
                else:
                    confusion[key] += 1

        return len(gt), errs, total_sync_errs, confusion, (gt, pred)

    @staticmethod
    def evaluate_single_list(eval_results):
        # sum all errors up
        all_eval = []
        total_instances = 0
        total_chars = 0
        total_char_errs = 0
        confusion = {}
        total_sync_errs = 0
        for chars, char_errs, sync_errs, conf, gt_pred in eval_results:
            total_instances += 1
            total_chars += chars
            total_char_errs += char_errs
            total_sync_errs += sync_errs
            for key, value in conf.items():
                if key not in confusion:
                    confusion[key] = value
                else:
                    confusion[key] += value

        # Note the sync errs can be higher than the true edit distance because
        # replacements are counted as 1
        # e.g. ed(in ewych, ierg ch) = 5
        #      sync(in ewych, ierg ch) = [{i: i}, {n: erg}, {ewy: }, {ch: ch}] = 6

        return {
            "single": all_eval,
            "total_instances": total_instances,
            "avg_ler": total_char_errs / total_chars,
            "total_chars": total_chars,
            "total_char_errs": total_char_errs,
            "total_sync_errs": total_sync_errs,
            "confusion": confusion,
        }

class TextExperimenter(Experimenter):
    @classmethod
    def print_results(cls, args, results, log):
        confusion = {}
        for result in results:
            for k, v in result['confusion'].items():
                confusion[k] = confusion.get(k, 0) + v

        r = np.array([(result['total_instances'], result['avg_ler'], result['total_chars'],
                       result['total_char_errs'], result['total_sync_errs'],
                       result['total_syllables'], result['avg_ser'], result['total_words'], result['avg_wer'],
                       result['confusion_count'], result['confusion_err'],
                       ) for result in results])

        pt = PrettyTable(['GT', 'PRED', 'Count'])
        for k, v in sorted(confusion.items(), key=lambda x: -x[1]):
            pt.add_row(k + (v, ))
        log.info(pt)

        pt = PrettyTable(['#', 'avg_ler', '#chars', '#errs', '#sync_errs', '#sylls', 'avg_ser', '#words', "avg_wer", "#conf", 'conf_err'])
        pt.add_row(np.mean(r, axis=0))
        pt.add_row(np.std(r, axis=0))
        log.info(pt)

        if args.magic_prefix:
            all_diffs = np.array(np.transpose([np.mean(r, axis=0), np.std(r, axis=0)])).reshape([-1])
            print("{}{}".format(args.magic_prefix, ','.join(map(str, list(all_diffs)))))
            return "{}{}".format(args.magic_prefix, ','.join(map(str, list(all_diffs))))

    def extract_gt_prediction(self, full_predictions: List[PredictionResult]):
        from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization
        lnp = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.SYLLABLES))

        def format_gt(s):
            s = lnp.apply(s)
            return s

        def flatten(x):
            return sum(x, [])
        pred = [[tl.hyphenated for tl in p.text_lines] for p in full_predictions]
        gt = [[format_gt(tl.line.operation.text_line.sentence.text()) for tl in p.text_lines] for p in full_predictions]

        return flatten(gt), flatten(pred)

    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions: List[PredictionResult]):
        raise NotImplemented
        for pcgts, pred in zip(output_pcgts, predictions):
            pcgts.page.annotations = Annotations.from_json(pred.annotations.to_json(), pcgts.page)
    def output_debug_images(self, predictions):
        raise NotImplemented

    def evaluate(self, predictions: Tuple[List[str], List[str]], evaluation_params):

        gt, pred = predictions

        def edit_on_tokens(gt: List[str], pred: List[str]):
            return min(1, edit_distance(gt, pred)[0] / len(gt) if len(gt)> 0 else 0), len(gt)

        def sentence_to_syllable_tokens(s: Sentence) -> List[str]:
            return [syl.text for syl in s.syllables]

        def sentence_to_words(s: Sentence) -> List[str]:
            return s.text().replace('-', '').replace('~', '').split()

        def chars_only(s: str):
            return LyricsNormalizationProcessor(self.args.global_args.dataset_params.lyrics_normalization).apply(s)

        gt_sentence = [Sentence.from_string(s) for s in gt]
        pred_sentence = [Sentence.from_string(s) for s in pred]

        #ocr_eval = Evaluator(data=None)
        gt_data = [chars_only(s) for s in gt]
        pred_data = [chars_only(s) for s in pred]

        #result = [Evaluator.evaluate(gt_data=[chars_only(s) for s in gt], pred_data=[chars_only(s) for s in pred])
        result = [Evaluator.evaluate_single(gt=i[0], pred=i[1]) for i in zip(gt_data, pred_data)]
        result = Evaluator.evaluate_single_list(eval_results=result)
        syllable_result = [edit_on_tokens(sentence_to_syllable_tokens(a), sentence_to_syllable_tokens(b)) for a, b in zip(gt_sentence, pred_sentence)]
        words_result = [edit_on_tokens(sentence_to_words(a), sentence_to_words(b)) for a, b in zip(gt_sentence, pred_sentence)]
        result['total_syllables'] = sum([n for _, n in syllable_result])
        result['avg_ser'] = np.mean([n for n, _ in syllable_result])
        result['total_words'] = sum([n for _, n in words_result])
        result['avg_wer'] = np.mean([n for n, _ in words_result])
        result['confusion_count'] = sum([c for k, c in result['confusion'].items()])
        result['confusion_err'] = result['confusion_count'] / result['total_syllables']

        pt = PrettyTable(['#', 'avg_ler', '#chars', '#errs', '#sync_errs', '#ser', 'avg_ser', '#words', 'avg_wer', '#confusions', '#confusions/sylls'])
        pt.add_row([result['total_instances'], result['avg_ler'], result['total_chars'], result['total_char_errs'], result['total_sync_errs'],
                    result['total_syllables'], result['avg_ser'], result['total_words'], result['avg_wer'],
                    result['confusion_count'], result['confusion_err'],
                    ]
                   )
        self.fold_log.debug(pt)

        return result
