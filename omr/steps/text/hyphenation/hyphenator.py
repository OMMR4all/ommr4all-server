import enum
from abc import ABC, abstractmethod
from typing import Dict

import dataclasses

from omr.dataset.dataset import LyricsNormalizationParams, LyricsNormalizationProcessor, LyricsNormalization
from loguru import logger


class HyphenDicts(enum.Enum):
    classic = "classic"
    classic_ec = "classic_ec"
    liturgical = "liturgical"
    modern = "modern"

    def get_internal_file_path(self):
        from ommr4all.settings import BASE_DIR
        import os
        path = os.path.join(BASE_DIR, 'internal_storage', 'hyphen_dictionaries')

        return os.path.join(path, {"classic": "hyph_la_classical.dic",
                                   "classic_ec": "hyph_la_classical_ec.dic",
                                   "liturgical": "hyph_la_liturgical.dic",
                                   "modern": "hyph_la_modern.dic"}[self.value])


class Hyphenator(ABC):
    def __init__(self, word_separator=' '):
        self.word_separator = word_separator

    @abstractmethod
    def apply_to_word(self, word: str):
        return word

    def apply_to_sentence(self, s: str):
        return self.word_separator.join(map(self.apply_to_word, s.split(self.word_separator)))


class CombinedHyphenator(Hyphenator):
    from database.start_up.load_text_variants_in_memory import syllable_dictionary

    def __init__(self, lang='la', left=2, right=2):
        super().__init__()
        from thirdparty.pyphen import Pyphen
        self.pyphen = Pyphen(filename=lang, lang=lang, left=left, right=right)
        self.dictionary = syllable_dictionary

    def apply_to_word(self, word: str):
        l_word = word.lower()
        if self.dictionary is not None and l_word in self.dictionary:
            hyphenated = self.dictionary[l_word]

            # correct lower/upper chars
            def correct_l_u_case(hyph_word: str, o_word: str):
                syl_indexes = [i for i, c in enumerate(hyph_word) if c == "-"]
                for i in syl_indexes:
                    o_word = o_word[:i] + "-" + o_word[i:]

                return o_word
            hyphenated = correct_l_u_case(hyph_word=hyphenated, o_word=word)

            if False or hyphenated != self.pyphen.inserted(word):
                logger.info("Hyhenation db: {} Grammar: {}".format(hyphenated, self.pyphen.inserted(word)))
            return hyphenated

        else:
            return self.pyphen.inserted(word)


class Pyphenator(Hyphenator):
    def __init__(self, lang='la', left=2, right=2):
        super().__init__()
        from thirdparty.pyphen import Pyphen
        self.pyphen = Pyphen(filename=lang, lang=lang, left=left, right=right)

    def apply_to_word(self, word: str):
        return self.pyphen.inserted(word)


class HyphenatorFromDictionary(Hyphenator):
    def __init__(self, words: Dict[str, str] = None, dictionary: str = None,
                 normalization: LyricsNormalizationParams = None):
        super().__init__()
        self.words = words if words else {}
        if normalization:
            normalization = LyricsNormalizationParams(**normalization.to_dict())
            normalization.lyrics_normalization = LyricsNormalization.SYLLABLES
            p = LyricsNormalizationProcessor(normalization)
        if dictionary:
            with open(dictionary) as f:
                for line in f:
                    word, hyphen = line.split()
                    if p:
                        word = p.apply(word)
                        hyphen = p.apply(hyphen)
                    self.words[word] = hyphen

        if len(self.words) == 0:
            raise Exception("Empty dictionary for hyphenation. Either pass the hyphenation directly or as a file")

    def apply_to_word(self, word: str):
        return self.words[word]


if __name__ == "__main__":
    from database import DatabaseBook
    from database.database_dictionary import DatabaseDictionary
    import os
    import django
    import os

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    from database.start_up.load_text_variants_in_memory import lyrics, syllable_dictionary

    #db = DatabaseDictionary.load(book=DatabaseBook("mulhouse_mass_transcription"))
    #database_hyphen_dictionary = db.to_hyphen_dict()


    def test(hyphenators, word):
        for ind, i in enumerate(hyphenators):
            print(f'{i.apply_to_word(word)}')


    HyphenDicts.classic.get_internal_file_path()

    # hyphen1 = CombinedHyphenator(lang=os.path.join(path, "hyph_la.dic"), left=1, right=1, dictionary=
    # database_hyphen_dictionary)
    #hyphen2 = CombinedHyphenator(lang=HyphenDicts.classic.get_internal_file_path(), left=1, right=1, dictionary=
    #database_hyphen_dictionary)
    hyphen3 = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1, right=1, dictionary=
    syllable_dictionary)
    #hyphen4 = CombinedHyphenator(lang=HyphenDicts.modern.get_internal_file_path(), left=1, right=1, dictionary=
    #database_hyphen_dictionary)
    hyphenators = [hyphen3]

    test(hyphenators, "domino")
