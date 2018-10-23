from . import Coords, MusicLine, SymbolType, MusicLines
from uuid import uuid4


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

    def neume_by_id(self, id):
        for ml in self.staffs:
            for neume in ml.symbols:
                if neume.symbol_type == SymbolType.NEUME and neume.id == id:
                    return neume

        return None
