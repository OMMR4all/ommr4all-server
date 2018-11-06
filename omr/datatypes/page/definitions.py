from enum import Enum
from typing import NamedTuple
import numpy as np


class AABB(NamedTuple):
    tl: np.ndarray
    br: np.ndarray


class EquivIndex(Enum):
    GROUND_TRUTH = 0
    CORRECTED = 1
    AI = 2
