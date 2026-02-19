from typing import NamedTuple

import albumentations
import cv2
from albumentations import RandomScale, RandomGamma, RandomBrightnessContrast, OneOf, ToGray, CLAHE, Compose, Affine, \
    ShiftScaleRotate

import albumentations as albu




def remove_nones(x):
    return [y for y in x if y is not None]


def default_transform():
    from segmentation.preprocessing.workflow import BinarizeDoxapy

    result = Compose([
        #RandomScale(),
        ShiftScaleRotate(rotate_limit=2, scale_limit=(-0.1, 0.1), shift_limit_x=0.2, shift_limit_y=0.2, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), mask_value=0),
        Affine(shear=2, cval=(255,255,255), cval_mask=0),
        #albumentations.HorizontalFlip(p=0.25),
        RandomGamma(),
        RandomBrightnessContrast(),
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

    ], additional_targets={'add_symbols_mask': 'mask'} )
    return result
def symbol_transform():
    from segmentation.datasets.dataset import compose

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
    augmentation = None
    architecture: str = 'unet'
    encoder: str = 'efficientnet-b3'
    custom_model: bool = False
    predefined_encoder_depth = 5
    predefined_decoder_channel = (256, 128, 64, 32, 16) #(256, 256, 196, 128, 64) #(256, 128, 64, 32, 16) #(256, 256, 196, 128, 64)
    use_batch_norm_layer = True
    custom_model_encoder_filter = [16, 32, 64, 256, 512]
    custom_model_decoder_filter = [16, 32, 64, 256, 512]
    custom_model_encoder_depth = 4
    additional_number_of_heads: int = 0

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
