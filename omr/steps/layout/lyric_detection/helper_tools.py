import enum
from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Dither


class ColorTypes(enum.Enum):
    red = enum.auto()
    blue = enum.auto()
    black = enum.auto()
    background = enum.auto()

    def get_aligned_color(self):
        return {1: [255, 0, 0],
                2: [0, 255, 0],
                3: [0, 0, 0],
                4: [186, 178, 157]
                }[self.value]
        pass

class ColorLabel:
    def __init__(self, r: int, g: int, b: int, label=ColorTypes.red):
        self.red = r
        self.blue = b
        self.green = g
        self.label = label

    def get_color_as_rgb_list(self):
        return [self.red, self.green, self.blue]
        pass


class Palette():
    def __init__(self, palette: List[ColorLabel]):
        self.palette = palette

    def to_pil_color_palette(self):
        palette = []
        for i in self.palette:
            palette += i.get_color_as_rgb_list()
        return palette


class Color_Palett_Reduction:
    def __init__(self, palette: Palette = None):
        if palette is None:
            palette = Palette([ColorLabel(186, 178, 157, ColorTypes.background),
                               ColorLabel(64, 60, 49, ColorTypes.black),
                               ColorLabel(109, 108, 103, ColorTypes.black),
                               ColorLabel(141, 87, 87, ColorTypes.red),
                               ColorLabel(139, 81, 69, ColorTypes.red),
                               ColorLabel(115, 93, 82, ColorTypes.red),
                               ColorLabel(177, 125, 104, ColorTypes.red),
                               ColorLabel(177, 75, 70, ColorTypes.red)])
        self.palette = palette
        self.palimage = Image.new('P', (16, 16))
        self.palimage.putpalette(self.palette.to_pil_color_palette() * 32)
        pass

    def quantize(self, image: Image):
        '''
                :param method: :data:`Quantize.MEDIANCUT` (median cut),
                       :data:`Quantize.MAXCOVERAGE` (maximum coverage),
                       :data:`Quantize.FASTOCTREE` (fast octree),
                       :data:`Quantize.LIBIMAGEQUANT` (libimagequant; check support
        '''
        return image.quantize(palette=self.palimage, dither=Dither.NONE,
                              colors=len(self.palette.to_pil_color_palette()), )

    def reduced_quatnize(self, image: Image):
        image = self.quantize(image).convert("RGB")
        np_array = np.array(image)
        for i in self.palette.palette:
            colors = i.get_color_as_rgb_list()

            a = np.all(np_array == colors, axis=-1, keepdims=True)
            np_array = np.where(a, i.label.get_aligned_color(), np_array)
        return Image.fromarray(np_array.astype('uint8'), 'RGB')
