from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from omr.datatypes import MusicLines, StaffLine, StaffLines, MusicLine
import cv2
from scipy.misc import imresize
from scipy.ndimage import distance_transform_edt
from skimage.morphology import watershed

def content_rect(binary, data_density_low, data_density_high):
    s = 10
    b = binary.astype(np.float)
    b = imresize(b, 1 / s, interp='bilinear')

    b1 = b > data_density_low
    b2 = b < data_density_high

    b = 1 - b1 & b2

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 * b.astype(np.uint8), 8, cv2.CV_32S)

    out = []
    for l in range(1, num_labels):
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        a = stats[l, cv2.CC_STAT_AREA]

        #if w < b.shape[1] / 5 or h < b.shape[0] / 5:
        #    continue

        out.append((l, np.sum(b == l)))

    out.sort(key=lambda x: x[1])

    out = (b == out[-1][0])

    y = out.sum(axis=1)
    x = out.sum(axis=0)

    d = distance_transform_edt(out)

    y = d.sum(axis=1)
    t = np.argmax(y[:len(y) // 2])
    b = np.argmax(y[len(y) // 2:]) + len(y) // 2

    x = d.sum(axis=0)
    l = np.argmax(x[:len(x) // 2])
    r = np.argmax(x[len(x) // 2:]) + len(x) // 2

    return t * s, b * s, l * s, r * s


def crop_images(binary, images):
    b = 1 - np.array(binary) / 255
    shape = b.shape
    center = b[shape[0] // 4: shape[0] // 4 * 3, shape[1] // 4: shape[1] // 4 * 3]
    data_density = np.mean(center)

    t, b, l, r = content_rect(b, data_density / 2, data_density * 2)

    return tuple([i.crop((l, t, r, b))for i in images]), (l, t, r, b)


if __name__ == '__main__':
    from gregorian_annotator_server.settings import PRIVATE_MEDIA_ROOT
    import os
    binary = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'deskewed_binary.png'))
    gray = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'deskewed_gray.jpg'))
    original = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'deskewed_original.jpg'))

    (binary, gray, original), r = crop_images(binary, (binary, gray, original))

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(binary)
    ax[1].imshow(gray)
    ax[2].imshow(original)

    plt.show()
