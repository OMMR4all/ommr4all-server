from typing import NamedTuple
from ocr4all_pixel_classifier.lib.trainer import AugmentationSettings
from ocr4all_pixel_classifier.lib.model import Architecture


class PageSegmentationTrainerParams(NamedTuple):
    data_augmentation: bool = False
    augmentation_settings: AugmentationSettings = AugmentationSettings(
        rotation_range=2.5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.01,
        zoom_range=[0.9, 1.1],
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=None,
        image_fill_mode='constant',
        image_cval=0,
        binary_fill_mode='constant',
        binary_cval=255,
        mask_fill_mode='constant',
        mask_cval=0,
    )
    architecture: Architecture = Architecture.FCN_SKIP
