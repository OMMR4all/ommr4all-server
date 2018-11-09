from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Any, Optional
import numpy as np
from omr.datatypes import Point, MusicLine, Page, MusicRegion
from dataclasses import dataclass
from copy import copy


class ImageData(NamedTuple):
    image: np.ndarray
    nearest_neighbour_rescale: bool


@dataclass
class ImageDataInput:
    images: List[ImageData]

    page: Optional[Page]
    music_region: Optional[MusicRegion]
    music_line: Optional[MusicLine]

    def __iter__(self):
        return self.images.__iter__()


class OperationOutput(NamedTuple):
    data: ImageDataInput
    params: Any


class ImageOperation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply_single(self, data: ImageDataInput) -> OperationOutput:
        pass

    def local_to_global_pos(self, p: Point, params: Any) -> Point:
        return p


class ImageOperationList(ImageOperation):
    def __init__(self, operations: List[ImageOperation]):
        super().__init__()
        self.operations: List[ImageOperation] = operations

    def apply_single(self, data: ImageDataInput) -> OperationOutput:
        data = copy(data)
        params = []
        for op in self.operations:
            op_out = op.apply_single(data)
            data = op_out.data
            params.append(op_out.params)

        return OperationOutput(data, params)

    def local_to_global_pos(self, p: Point, params: List[Any]):
        for op, param in zip(reversed(self.operations), reversed(params)):
            p = op.local_to_global_pos(p, param)



