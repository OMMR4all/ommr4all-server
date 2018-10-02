from . import Coords, TextLine, TextEquiv
from typing import List
from enum import Enum


class TextRegionType(Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LYRICS = "lyrics"


class TextRegion:
    def __init__(self,
                 region_type=TextRegionType.PARAGRAPH,
                 coords=Coords(),
                 text_lines: List[TextLine]=list(),
                 text_equivs: List[TextEquiv]=list()):
        self.region_type = region_type
        self.coords = coords
        self.text_lines = text_lines
        self.text_equivs = text_equivs

    def syllable_by_id(self, syllable_id):
        if self.region_type != TextRegionType.LYRICS:
            return None

        for t in self.text_equivs + self.text_lines:
            r = t.syllable_by_id(syllable_id)
            if r:
                return r

        return None

    def _resolve_cross_refs(self, page):
        pass

    @staticmethod
    def from_json(json: dict):
        return TextRegion(
            TextRegionType(json.get('type', TextRegionType.PARAGRAPH)),
            Coords.from_json(json.get('coords', [])),
            [TextLine.from_json(l) for l in json.get('textLines', [])],
            [TextEquiv.from_json(t) for t in json.get('textEquivs', [])],
        )

    def to_json(self):
        return {
            'type': self.region_type.value,
            'coords': self.coords.to_json(),
            'textLines': [l.to_json for l in self.text_lines],
            'textEquivs': [t.to_json for t in self.text_equivs],
        }
