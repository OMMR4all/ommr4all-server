from typing import List
from .syllable import Syllable


class Word:
    def __init__(self,
                 syllables: List[Syllable]):
        self.syllables = syllables

    @staticmethod
    def from_json(json: dict):
        return Word(
            [Syllable.from_json(s) for s in json['syllables']]
        )

    def to_json(self):
        return {
            'syllables': [s.to_json() for s in self.syllables]
        }