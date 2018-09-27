import cv2
from collections import namedtuple
import numpy as np

ConnectedComponents = namedtuple('ConnectedComponents', ['num_labels', 'labels', 'stats', 'centroids'])


def connected_compontents_with_stats(binary: np.ndarray):
    return ConnectedComponents(*cv2.connectedComponentsWithStats(255 - binary, 8, cv2.CV_32S))

