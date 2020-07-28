from typing import NamedTuple
from segmentation.modules import Architecture
from segmentation.dataset import compose
import albumentations as albu


def symbol_transform():
    result = [
        albu.HorizontalFlip(),
        albu.Rotate((-2, 2), border_mode=0, value=0),
        albu.RandomGamma(),
        albu.RandomBrightnessContrast(0.1, 0.1),
        albu.RandomScale(),
    ]
    return compose([result])


class PageSegmentationTrainerTorchParams(NamedTuple):
    data_augmentation: bool = False
    augmentation = symbol_transform()
    architecture: Architecture = Architecture.UNET
    encoder: str = 'efficientnet-b3'

    # augmentation_settings: AugmentationSettings = AugmentationSettings(
    #    rotation_range=2.5,
    #    width_shift_range=0.2,
    #    height_shift_range=0.2,
    #    shear_range=0.01,
    #    zoom_range=[0.9, 1.1],
    #    horizontal_flip=False,
    #    vertical_flip=False,
    #    brightness_range=None,
    #    image_fill_mode='constant',
    #    image_cval=0,
    #    binary_fill_mode='constant',
    #    binary_cval=255,
    #    mask_fill_mode='constant',
    #    mask_cval=0,
    # )
