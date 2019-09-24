from typing import NamedTuple
from pagesegmentation.lib.trainer import AugmentationSettings
from pagesegmentation.lib.model import Architecture


class PageSegmentationTrainerParams(NamedTuple):
    data_augmentation: bool = False
    augmentation_settings: AugmentationSettings = AugmentationSettings(
        rotation_range=0.1,
        width_shift_range=0.025,
        height_shift_range=0.025,
        shear_range=0.01,
        zoom_range=[0.99, 1.01],
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=[1, 1],
    )
    architecture: Architecture = Architecture.FCN_SKIP
