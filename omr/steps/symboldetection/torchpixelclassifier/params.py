from typing import NamedTuple

import albumentations

from segmentation.modules import Architecture
from segmentation.dataset import compose
import albumentations as albu

from segmentation.preprocessing.workflow import BinarizeDoxapy
from segmentation.settings import PredefinedNetworkSettings, CustomModelSettings


def remove_nones(x):
    return [y for y in x if y is not None]


def default_transform():
    result = albumentations.Compose([
        albumentations.RandomScale(),
        #albumentations.HorizontalFlip(p=0.25),
        albumentations.RandomGamma(),
        albumentations.RandomBrightnessContrast(),
        albumentations.OneOf([
            albumentations.OneOf([
                BinarizeDoxapy("sauvola"),
                BinarizeDoxapy("ocropus"),
                BinarizeDoxapy("isauvola"),
            ]),
            albumentations.OneOf([
                albumentations.ToGray(),
                albumentations.CLAHE()
            ])
        ], p=0.3)
    ])
    return result
def symbol_transform():
    result = [
        # albu.HorizontalFlip(),
        albu.Rotate((-2, 2), border_mode=0, value=0),
        albu.RandomGamma(),
        albu.RandomBrightnessContrast(0.1, 0.1),
        albu.RandomScale(),
    ]
    return compose([result])


class PageSegmentationTrainerTorchParams(NamedTuple):
    data_augmentation: bool = True
    augmentation = default_transform()
    architecture: Architecture = Architecture.UNET
    encoder: str = 'efficientnet-b3'
    custom_model: bool = False
    predefined_encoder_depth = PredefinedNetworkSettings.encoder_depth
    predefined_decoder_channel = PredefinedNetworkSettings.decoder_channel

    custom_model_encoder_filter = [16, 32, 64, 256, 512]
    custom_model_decoder_filter = [16, 32, 64, 256, 512]
    custom_model_encoder_depth = CustomModelSettings.encoder_depth

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
