from . import Coords, MusicLine
from typing import List
from uuid import uuid4


class MusicRegion:
    def __init__(self,
                 mr_id: str = None,
                 coords: Coords = None,
                 staffs: List[MusicLine]=None):
        self.id = mr_id if mr_id else str(uuid4())
        self.coords = coords if coords else Coords()
        self.staffs = staffs if staffs else []

    @staticmethod
    def from_json(json):
        return MusicRegion(
            json.get('id', None),
            Coords.from_json(json.get('coords', [])),
            [MusicLine.from_json(s) for s in json.get('musicLines', [])],
        )

    def to_json(self):
        return {
            "id": self.id,
            "coords": self.coords.to_json(),
            "musicLines": [s.to_json() for s in self.staffs],
        }

    def neume_by_id(self, id):
        for ml in self.staffs:
            for neume in ml.neumes:
                if neume.id == id:
                    return neume

        return None
