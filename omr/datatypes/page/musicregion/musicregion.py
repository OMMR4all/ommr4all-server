from . import Coords, StaffEquiv
from typing import List


class MusicRegion:
    def __init__(self, coords=Coords(), staffs: List[StaffEquiv]=list()):
        self.coords = coords
        self.staffs = staffs

    def _resolve_cross_refs(self, page):
        for staff in self.staffs:
            staff._resolve_cross_refs(page)

    @staticmethod
    def from_json(json):
        return MusicRegion(
            Coords.from_json(json.get('coords', [])),
            [StaffEquiv.from_json(s) for s in json.get('staffEquivs', [])],
        )

    def to_json(self):
        return {
            "coords": self.coords.to_json(),
            "staffEquivs": [s.to_json() for s in self.staffs],
        }

    def has_staff_equiv_by_index(self, index):
        return len([s for s in self.staffs if s.index == index]) > 0

    def staff_equiv_by_index(self, index):
        for s in self.staffs:
            if s.index == index:
                return s
        return None
