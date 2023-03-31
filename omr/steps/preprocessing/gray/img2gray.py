import numpy as np
from PIL import Image
import logging


def im2gray(img: Image) -> Image:
    ni = np.array(img).astype(float)
    if ni.ndim == 3:
        return Image.fromarray(((ni[:, :, 2] + ni[:, :, 1] + ni[:, :, 0]) / 3).astype(np.uint8))

    elif ni.ndim == 2:
        return img

    else:
        logging.error("Image has invalid dimensions ndim == {}".format(img.ndim))
        return img
