from typing import NamedTuple


class CalamariParams(NamedTuple):
    network: str = 'cnn=32:3x3,pool=2x2,cnn=64:3x3,pool=1x2,cnn=64:3x3,lstm=100,dropout=0.5'
