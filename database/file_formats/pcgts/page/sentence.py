from typing import List
from .syllable import Syllable, SyllableConnection


class Sentence:
    def __init__(self,
                 syllables: List[Syllable]):
        self.syllables = syllables

    def text(self):
        t = ''
        for syllable in self.syllables:
            if syllable.connection == SyllableConnection.NEW:
                if len(t) == 0:
                    t += syllable.text
                else:
                    t += ' ' + syllable.text
            elif syllable.connection == SyllableConnection.VISIBLE:
                t += '~' + syllable.text
            else:
                t += '-' + syllable.text
        return t

    def syllable_by_id(self, syllable_id):
        for s in self.syllables:
            if s.id == syllable_id:
                return s

        return None

    @staticmethod
    def from_json(json: list):
        return Sentence(
            [Syllable.from_json(s) for s in json]
        )

    def to_json(self):
        return [s.to_json() for s in self.syllables]

