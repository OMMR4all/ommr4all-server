from . import Coords
from typing import List
from uuid import uuid4
from .sentence import Sentence


class TextLine:
    def __init__(self,
                 tl_id: str = None,
                 coords: Coords = None,
                 sentence: Sentence = None,
                 reconstructed=False,
                 ):
        self.id = tl_id if tl_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.sentence = sentence if sentence else Sentence([])
        self.reconstructed = reconstructed

    def text(self, with_drop_capital=True):
        return self.sentence.text(with_drop_capital=with_drop_capital)

    def syllable_by_id(self, syllable_id):
        return self.sentence.syllable_by_id(syllable_id)

    @staticmethod
    def from_json(json: dict):
        return TextLine(
            json.get('id', str(uuid4())),
            Coords.from_json(json.get('coords', [])),
            Sentence.from_json(json.get('syllables', [])),
            json.get('reconstructed', False),
        )

    def to_json(self):
        return {
            'id': self.id,
            'coords': self.coords.to_json(),
            'syllables': self.sentence.to_json(),
            'reconstructed': self.reconstructed,
        }