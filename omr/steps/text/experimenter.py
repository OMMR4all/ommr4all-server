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
        from calamari_ocr.ocr.evaluator import Evaluator
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
        for i, y in list(zip(gt_sentence, pred_sentence)):
            print(i.text())
            print(y.text())
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
