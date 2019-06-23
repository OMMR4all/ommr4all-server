from . import Coords, Line
from typing import List
from enum import Enum
from uuid import uuid4


class TextRegionType(Enum):
    PARAGRAPH = 0
    HEADING = 1
    LYRICS = 2
    DROP_CAPITAL = 3
    FOLIO_NUMBER = 4


class TextRegion:
    def __init__(self,
                 id: str = None,
                 region_type=TextRegionType.PARAGRAPH,
                 coords: Coords = None,
                 text_lines: List[Line] = None,
                 ):
        self.id = id if id else str(uuid4())
        self.region_type = region_type
        self.coords = coords if coords else Coords()
        self.text_lines = text_lines if text_lines else []

    @property
    def children(self):
        return self.text_lines

    def syllable_by_id(self, syllable_id, require=False):
        if self.region_type == TextRegionType.LYRICS:
            for t in self.text_lines:
                r = t.syllable_by_id(syllable_id)
                if r:
                    return r

        if require:
            raise ValueError("Syllable with ID {} not found in text region {}".format(syllable_id, self.id))

        return None

    def _resolve_cross_refs(self, page):
        pass

    @staticmethod
    def from_json(json: dict):
        return TextRegion(
            json.get('id', None),
            TextRegionType(json.get('type', TextRegionType.PARAGRAPH)),
            Coords.from_json(json.get('coords', [])),
            [TextLine.from_json(l) for l in json.get('textLines', [])],
        )

    def to_json(self):
        return {
            'id': self.id,
            'type': self.region_type.value,
            'coords': self.coords.to_json(),
            'textLines': [l.to_json() for l in self.text_lines],
        }
