from typing import NamedTuple, List, Optional


class CalamariParams(NamedTuple):
    network: str = 'cnn=32:3x3,pool=2x2,cnn=64:3x3,pool=1x2,cnn=64:3x3,lstm=100,dropout=0.5'
    n_folds: int = 0  # default = 0 means no folds
    single_folds: Optional[List[int]] = None
