from typing import NamedTuple, Optional
from pagesegmentation.lib.data_augmenter import DataAugmenterBase


class PageSegmentationTrainerParams(NamedTuple):
    data_augmenter: Optional[DataAugmenterBase] = None
