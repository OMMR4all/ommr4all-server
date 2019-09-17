from .region import Region, Coords
from .definitions import BlockType
from .line import Line
from .musicsymbol import SymbolType
from typing import List, Optional
from uuid import uuid4


class Block(Region):
    def __init__(self,
                 block_type: BlockType,
                 id: Optional[str] = None,
                 coords: Optional[Coords] = None,
                 lines: Optional[List[Line]] = None,
                 ):
        self.block_type = block_type
        self.lines = lines if lines else []
        super().__init__(id, coords)

    def _compute_aabb(self):
        aabb = super()._compute_aabb()
        for l in self.lines:
            aabb = aabb.union(l.aabb)

        return aabb

    @staticmethod
    def from_json(d: dict) -> 'Block':
        return Block(
            BlockType(d.get('type', BlockType.MUSIC.value)),
            d.get('id', str(uuid4())),
            Coords.from_json(d.get('coords', '')),
            [Line.from_json(l) for l in d.get('lines', [])],
        )

    def to_json(self) -> dict:
        return {
            **super().to_json(),
            **{
                'type': self.block_type.value,
                'lines': [l.to_json(self.block_type) for l in self.lines],
            }
        }

    def line_by_id(self, id: str) -> Optional[Line]:
        for l in self.lines:
            if l.id == id:
                return l
        return None

    def syllable_by_id(self, syllable_id, require=False):
        if self.block_type == BlockType.LYRICS:
            for l in self.lines:
                r = l.syllable_by_id(syllable_id)
                if r:
                    return r

        if require:
            raise ValueError("Syllable with ID {} not found in block {}".format(syllable_id, self.id))

        return None

    def note_by_id(self, id, required=False):
        for ml in self.lines:
            for s in ml.symbols:
                if s.symbol_type == SymbolType.NOTE and s.id == id:
                    return s

        if required:
            raise ValueError("Note with ID {} not found in music region".format(id, self.id))

        return None

    def update_note_names(self, current_clef=None):
        for line in self.lines:
            current_clef = line.update_note_names(current_clef)

        return current_clef

    def rotate(self, degree, origin):
        for line in self.lines:
            self.coords.rotate(degree, origin)
            line.rotate(degree, origin)

