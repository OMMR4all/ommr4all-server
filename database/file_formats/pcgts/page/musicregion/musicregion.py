from . import Coords, MusicLine, SymbolType, MusicLines
from uuid import uuid4
import numpy as np


class MusicRegion:
    def __init__(self,
                 mr_id: str = None,
                 coords: Coords = None,
                 staffs: MusicLines = None):
        self.id = mr_id if mr_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.staffs = staffs if staffs else MusicLines()
        assert(isinstance(self.staffs, MusicLines))
        assert(isinstance(self.coords, Coords))

    @property
    def children(self):
        return self.staffs

    @staticmethod
    def from_json(json):
        return MusicRegion(
            json.get('id', None),
            Coords.from_json(json.get('coords', [])),
            MusicLines.from_json(json.get('musicLines', [])),
        )

    def to_json(self):
        return {
            "id": self.id,
            "coords": self.coords.to_json(),
            "musicLines": self.staffs.to_json(),
        }

    def neume_by_id(self, id, required=False):
        for ml in self.staffs:
            for neume in ml.symbols:
                if neume.symbol_type == SymbolType.NEUME and neume.id == id:
                    return neume

        if required:
            raise ValueError("Neume with ID {} not found in music region".format(id, self.id))

        return None

    def extract_music_line_images_and_gt(self, page: np.ndarray) -> (MusicLine, np.ndarray, str):
        for r in self.staffs.extract_images_and_gt(page):
            yield r
