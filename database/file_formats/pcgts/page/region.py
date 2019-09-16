from .coords import Coords, Rect
from typing import Optional
from uuid import uuid4


class Region:
    def __init__(self,
                 id: Optional[str] = None,
                 coords: Optional[Coords] = None,
                 ):
        self.id = id if id else str(uuid4())
        self.coords: Coords = coords if coords else Coords()
        self.aabb: Rect = self._compute_aabb()

    def _compute_aabb(self):
        return self.coords.aabb()

    def to_json(self):
        return {
            'id': self.id,
            'coords': self.coords.to_json(),
        }
