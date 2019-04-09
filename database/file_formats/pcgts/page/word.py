from typing import List
from .syllable import Syllable, SyllableConnection


class Word:
    def __init__(self,
                 syllables: List[Syllable]):
        self.syllables = syllables

    def text(self):
        t = ''
        for syllable in self.syllables:
            if syllable.connection == SyllableConnection.NEW:
                t += syllable.text
            elif syllable.connection == SyllableConnection.VISIBLE:
                t += '~' + syllable.text
            else:
                t += '-' + syllable.text
        return t

    @staticmethod
    def from_json(json: dict):
        return Word(
            [Syllable.from_json(s) for s in json['syllables']]
        )

    def to_json(self):
        return {
            'syllables': [s.to_json() for s in self.syllables]
        }


class Sentence:
    def __init__(self,
                 words: List[Word]):
        self.words = words

    def text(self):
        if len(self.words) == 0:
            return ""
        return " ".join([w.text() for w in self.words])

    def syllable_by_id(self, syllable_id):
        for w in self.words:
            for s in w.syllables:
                if s.id == syllable_id:
                    return s

        return None

    @staticmethod
    def from_json(json: list):
        return Sentence(
            [Word.from_json(s) for s in json]
        )

    def to_json(self):
        return [
            w.to_json() for w in self.words
        ]

