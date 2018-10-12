from . import Coords, MusicLine
from typing import List


class MusicRegion:
    def __init__(self, coords=Coords(), staffs: List[MusicLine]=None):
        self.coords = coords
        self.staffs = staffs if staffs else []

    @staticmethod
    def from_json(json):
        return MusicRegion(
            Coords.from_json(json.get('coords', [])),
            [MusicLine.from_json(s) for s in json.get('musicLines', [])],
        )

    def to_json(self):
        return {
            "coords": self.coords.to_json(),
            "musicLines": [s.to_json() for s in self.staffs],
        }

    def has_staff_equiv_by_index(self, index):
        return len([s for s in self.staffs if s.index == index]) > 0

    def staff_equiv_by_index(self, index):
        for s in self.staffs:
            if s.index == index:
                return s
        return None
