from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CalamariParams:
    network: str = 'cnn=40:3x3,pool=2x2,db=60:3:3x3,pool=2x2,tcnn=40:4x4,concat=3:-1,tcnn=40:4x4,db=80:4:3x3,dropout=0.5'
    n_folds: int = 0  # default = 0 means no folds
    single_folds: Optional[List[int]] = None
    channels: int = 1
