import numpy as np


def normalize_raw_image(raw):
    image = raw.astype(np.float32) - np.amin(raw)
    image /= np.amax(raw)
    return image

