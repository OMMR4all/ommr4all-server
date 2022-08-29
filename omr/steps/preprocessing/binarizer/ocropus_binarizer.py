import numpy as np
from scipy.ndimage import filters, interpolation, morphology
from skimage.transform import resize
import scipy.stats as stats
from PIL import Image
from omr.image_util import normalize_raw_image

from .binarize import Binarizer


def estimate_local_whitelevel(image, zoom=0.5, perc=80, range=20, debug=0):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    m = interpolation.zoom(image,zoom)
    m = filters.percentile_filter(m,perc,size=(range,2))
    m = filters.percentile_filter(m,perc,size=(2,range))
    m = interpolation.zoom(m,1.0/zoom)
    if m.shape != image.shape:
        m = resize((m * 255).astype(np.uint8), image.shape, preserve_range=True) / 255

    w,h = np.minimum(np.array(image.shape),np.array(m.shape))
    flat = np.clip(image[:w,:h]-m[:w,:h]+1,0,1)

    if np.isnan(flat).any():
        raise ValueError()

    return flat


def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
    '''# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0,d1 = flat.shape
    o0,o1 = int(bignore*d0),int(bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*np.amax(v))
        v = morphology.binary_dilation(v,structure=np.ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=np.ones((1,int(e*50))))
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),lo)
    hi = stats.scoreatpercentile(est.ravel(),hi)
    return lo, hi


def binarize(image):
    flat = estimate_local_whitelevel(image)
    lo, hi = estimate_thresholds(flat)

    flat -= lo
    flat /= (hi-lo)
    flat = np.clip(flat, 0, 1)

    assert(image.shape == flat.shape)
    return flat > 0.5


class OCRopusBin(Binarizer):
    def __init__(self):
        super().__init__()

    def binarize(self, image: Image):
        return Image.fromarray(binarize(normalize_raw_image(np.array(image.convert('L')))).astype(np.uint8) * 255)

    def binarize_from_array(self, image: np.array):
        pil_image = Image.fromarray(image)
        grayscale = pil_image.convert('L')
        return binarize(normalize_raw_image(np.array(grayscale))).astype(np.uint8) * 255