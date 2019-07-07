import numpy as np
from itertools import tee
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from typing import NamedTuple


class LineDistanceResult(NamedTuple):
    line_thickness: float
    white_space_distance: float
    line_distance: float


class LineDistanceComputer:

    def __init__(self, max_width=800, threshold = 0.2):
        self.max_width = max_width
        self.threshold = threshold

    def get_line_distance(self, binary_image: np.array) -> LineDistanceResult:
        o_height, o_width = binary_image.shape
        factor = o_width / self.max_width
        height = int(o_height / factor)
        image = 1 - binary_image
        image_resized = resize(image, (height, self.max_width), preserve_range=True, anti_aliasing=True, order=1) > self.threshold
        wr, br = np.array(vertical_runs(1 - image_resized)) * factor
        return LineDistanceResult(br, wr, br + wr)


def enhance(image):
    #image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 75, 75)
    image = (1 - np.clip(convolve2d(1 - image, np.full((1, 15), 0.2), mode="same"), 0, 255))
    distance = image.shape[1] / 20
    weight = gaussian_filter(image, sigma=(0, distance))
    return weight


def vertical_runs(img: np.array) -> [int, int]:
    img = np.transpose(img)
    h = img.shape[0]
    w = img.shape[1]
    transitions = np.transpose(np.nonzero(np.diff(img)))
    white_runs = [0] * (w + 1)
    black_runs = [0] * (w + 1)
    a, b = tee(transitions)
    next(b, [])
    for f, g in zip(a, b):
        if f[0] != g[0]:
            continue
        tlen = g[1] - f[1]
        if img[f[0], f[1] + 1] == 1:
            white_runs[tlen] += 1
        else:
            black_runs[tlen] += 1

    for y in range(h):
        x = 1
        col = img[y, 0]
        while x < w and img[y, x] == col:
            x += 1
        if col == 1:
            white_runs[x] += 1
        else:
            black_runs[x] += 1

        x = w - 2
        col = img[y, w - 1]
        while x >= 0 and img[y, x] == col:
            x -= 1
        if col == 1:
            white_runs[w - 1 - x] += 1
        else:
            black_runs[w - 1 - x] += 1

    black_r = np.argmax(black_runs) + 1
    # on pages with a lot of text the staffspaceheigth can be falsified.
    # --> skip the first elements of the array, we expect the staff lines distance to be at least twice the line height
    white_r = np.argmax(white_runs[black_r * 3:]) + 1 + black_r * 3
    return white_r, black_r


if __name__ == '__main__':
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import os
    from PIL import  Image
    binary = np.asarray(Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'pages', 'page00000001', 'binary_deskewed.png')), dtype='uint8') / 255
    binary1 = np.asarray(Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'pa_resized1419', 'pages', '00039r', 'binary_deskewed.png')), dtype='uint8') / 255

    gray = np.asarray(Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'pages', 'page00000001', 'gray_deskewed.jpg')))
    gray1 = np.asarray(Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'pa_resized1419', 'pages', '00037v', 'gray_deskewed.jpg')))
    print(LineDistanceComputer().get_line_distance(binary1))
