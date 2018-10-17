from . import Coords, TextEquiv
from typing import List
from uuid import uuid4
from .word import Word


class TextLine:
    def __init__(self,
                 tl_id: str = None,
                 coords: Coords = None,
                 text_equivs: List[TextEquiv] = None,
                 words: List[Word] = None,
                 ):
        self.id = tl_id if tl_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.text_equivs = text_equivs if text_equivs else []
        self.words = words if words else []

    def syllable_by_id(self, syllable_id):
        for w in self.words:
            for s in w.syllables:
                if s.id == syllable_id:
                    return s

        return None

    @staticmethod
    def from_json(json: dict):
        return TextLine(
            json.get('id', str(uuid4())),
            Coords.from_json(json.get('coords', [])),
            [TextEquiv.from_json(t) for t in json.get('textEquivs', [])],
            [Word.from_json(w) for w in json.get('words', [])]
        )

    def to_json(self):
        return {
            'id': self.id,
            'coords': self.coords.to_json(),
            'textEquivs': [t.to_json() for t in self.text_equivs],
            'words': [w.to_json() for w in self.words],
        }