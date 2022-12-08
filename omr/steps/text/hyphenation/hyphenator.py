from abc import ABC, abstractmethod
from typing import Dict

from omr.dataset.dataset import LyricsNormalizationParams, LyricsNormalizationProcessor, LyricsNormalization
from loguru import logger

class Hyphenator(ABC):
    def __init__(self, word_separator=' '):
        self.word_separator = word_separator

    @abstractmethod
    def apply_to_word(self, word: str):
        return word

    def apply_to_sentence(self, s: str):
        return self.word_separator.join(map(self.apply_to_word, s.split(self.word_separator)))


class CombinedHyphenator(Hyphenator):
    def __init__(self, lang='la', left=2, right=2, dictionary=None):
        super().__init__()
        from thirdparty.pyphen import Pyphen
        self.pyphen = Pyphen(lang=lang, left=left, right=right)
        self.dictionary = dictionary
    def apply_to_word(self, word: str):
        if self.dictionary is not None and word in self.dictionary:
            hyphenated = self.dictionary[word][0]
            logger.info("Hyhenation db: {} Grammar: {}".format(hyphenated, self.pyphen.inserted(word)))
            return hyphenated

        else:
            return self.pyphen.inserted(word)
class Pyphenator(Hyphenator):
    def __init__(self, lang='la', left=2, right=2):
        super().__init__()
        from thirdparty.pyphen import Pyphen
        self.pyphen = Pyphen(lang=lang, left=left, right=right)

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
