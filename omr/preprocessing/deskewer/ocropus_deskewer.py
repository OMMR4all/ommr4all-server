import numpy as np
from scipy.ndimage import interpolation
from PIL import Image

from omr.preprocessing.deskewer import Deskewer


def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a


def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    d0, d1 = flat.shape
    o0, o1 = int(bignore*d0), int(bignore*d1)  # border ignore
    flat = np.amax(flat)-flat
    flat -= np.amin(flat)
    est = flat[o0:d0-o0, o1:d1-o1]
    ma = maxskew
    ms = int(2*maxskew*skewsteps)
    angle = estimate_skew_angle(est,np.linspace(-ma, ma, ms+1))
    return angle


class OcropusDeskewer(Deskewer):
    def __init__(self,
                 binarizer
                 ):
        super().__init__(binarizer)

    def _estimate_skew_angle(self, color_image, gray_image, binary_image):
        return estimate_skew(np.array(binary_image))


if __name__ == '__main__':
    from omr.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import os
    binary = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'binary_original.png'))
    gray = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'gray_original.png'))
    original = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'color_original.jpg'))
    deskewer = OcropusDeskewer(OCRopusBin())
    deskewer.deskew(original, gray, binary)
    print(deskewer.angle)
    deskewer.plot()
