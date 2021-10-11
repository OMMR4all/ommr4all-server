import logging
import os
from database.database_book_documents import DatabaseBookDocuments
from database.database_dictionary import DatabaseDictionary
from database.file_formats import PcGts
from database.file_formats.pcgts import Line
from database.file_formats.pcgts.page import Sentence
from ommr4all import settings

from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    PredictionCallback, AlgorithmPredictionResultGenerator

from database import DatabasePage, DatabaseBook
from typing import List, Optional, NamedTuple
from omr.steps.text.correction_tools.dictionary_corrector.meta import Meta
from symspellpy import SymSpell

from omr.steps.text.hyphenation.hyphenator import Pyphenator

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()


class DictionaryCorrector(SymSpell):
    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7,
                 count_threshold=1):
        super().__init__(max_dictionary_edit_distance=max_dictionary_edit_distance, prefix_length=prefix_length,
                         count_threshold=count_threshold)
        self.sym_spell = SymSpell()
        self.book_id = None
        self.hyphen = Pyphenator()

    def segmentate_correct_and_hyphenate_text(self, text, hyphenate=True, edit_distance=2):
        sentence = self.word_segmentation(text, edit_distance)
        if hyphenate:
            sentence = self.hyphen.apply_to_sentence(sentence.corrected_string)
        return sentence

    def load_dict(self, book):

        if self.book_id != book.book:
            #self.sym_spell = SymSpell()
            self.book_id = book.book
            b = DatabaseDictionary.load(book)
            for entry in b.dictionary.freq_list:
                self.create_dictionary_entry(entry.word, entry.frequency)
            path = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_dictionary',
                                'bigram_default_dictionary.txt')
            loaded = self.load_bigram_dictionary(path, 0, 2)
            logger.info("Successfully loaded bigram dict" if loaded else "Failed to load bigram dict")

        pass


class PredictionResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class PredictionResultSingleLine(NamedTuple):
    line: Line
    hyphenated: str

    def to_dict(self):
        return {'sentence': self.hyphenated,
                'id': self.line.id,
                }


class PredictionResult(AlgorithmPredictionResult, NamedTuple, metaclass=PredictionResultMeta):
    pcgts: PcGts
    dataset_page: DatabasePage
    text_lines: List[PredictionResultSingleLine]

    def to_dict(self):
        return {'textLines': [l.to_dict() for l in self.text_lines]}

    def store_to_page(self):
        for line in self.text_lines:
            line.line.sentence = Sentence.from_string(line.hyphenated)
        self.pcgts.page.annotations.connections.clear()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class ResultMeta(NamedTuple.__class__, AlgorithmPredictionResult.__class__):
    pass


class Predictor(AlgorithmPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        self.dict_corrector = DictionaryCorrector()

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return True

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        book = pages[0].book

        self.dict_corrector.load_dict(book=book)

        for page in pages:
            pcgts = page.pcgts()
            text_lines: List[Line] = pcgts.page.all_text_lines()
            single_line_pred_result: List[PredictionResultSingleLine] = []
            for t_line in text_lines:
                text = t_line.text()
                text = text.replace("-", "")
                sentence = self.dict_corrector.segmentate_correct_and_hyphenate_text(text)
                single_line_pred_result.append(PredictionResultSingleLine(t_line, hyphenated=sentence))

            yield PredictionResult(pcgts, page, single_line_pred_result)


if __name__ == '__main__':
    from omr.dataset import DatasetParams, RegionLineMaskData
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    b = DatabaseBook('demo')
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[0:1]]
    pred = Predictor(AlgorithmPredictorSettings(Meta.best_model_for_book(b)))
    ps = list(pred.predict([p.page.location for p in val_pcgts]))
    import matplotlib.pyplot as plt
