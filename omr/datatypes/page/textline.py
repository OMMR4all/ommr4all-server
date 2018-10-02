from . import Coords, TextEquiv
from typing import List


class TextLine:
    def __init__(self, coords=Coords, text_equivs: List[TextEquiv]=[]):
        self.coords = coords
        self.text_equivs = text_equivs

    def syllable_by_id(self, syllable_id):
        for t in self.text_equivs:
            r = t.syllable_by_id(syllable_id)
            if r:
                return r

        return None

    @staticmethod
    def from_json(json: dict):
        return TextLine(
            Coords.from_json(json.get('coords', [])),
            [TextEquiv.from_json(t) for t in json.get('textEquivs', [])],
        )

    def to_json(self):
        return {
            'coords': self.coords.to_json(),
            'textEquivs': [t.to_json() for t in self.text_equivs]
        }