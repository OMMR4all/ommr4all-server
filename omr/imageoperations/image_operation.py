from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Any, Optional, Generator
import numpy as np
from database.file_formats.pcgts import Point, MusicLines, Page, MusicRegion, MusicLine
from dataclasses import dataclass
from copy import copy


@dataclass
class ImageData:
    image: np.ndarray
    nearest_neighbour_rescale: bool


@dataclass
class ImageOperationData:
    images: List[ImageData]
    params: Any = None

    page: Optional[Page] = None
    page_image: np.ndarray = None
    music_region: Optional[MusicRegion] = None
    music_line: Optional[MusicLine] = None

    def __iter__(self):
        return self.images.__iter__()


OperationOutput = List[ImageOperationData]


class ImageOperation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        pass

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        return p


class ImageOperationList(ImageOperation):
    def __init__(self, operations: List[ImageOperation]):
        super().__init__()
        self.operations = operations

    def apply_single(self, data: ImageOperationData) -> OperationOutput:
        data = [copy(data)]
        data[0].params = [data[0].params]
        for op in self.operations:
            out = []
            for d in data:
                for o in op.apply_single(copy(d)):
                    o.params = d.params + [o.params]
                    out.append(o)

            data = out

        return data

    def local_to_global_pos(self, p: Point, params: List[Any]) -> Point:
        for op, param in zip(reversed(self.operations), reversed(params)):
            p = op.local_to_global_pos(p, param)

        return p



