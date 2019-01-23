from omr.datatypes import StaffLines, StaffLine, MusicLine, Coords
from omr.preprocessing.util.connected_compontents import ConnectedComponents, connected_compontents_with_stats
import numpy as np
import json
from typing import List
import cv2
import matplotlib.pyplot as plt
from scipy import signal, spatial
from skimage.measure import approximate_polygon
from scipy.ndimage.filters import gaussian_filter

import itertools as IT
def scale_polygon(path,offset):
    center = centroid_of_polygon(path)
    for i in path:
        if i[0] > center[0]:
            i[0] += offset
        else:
            i[0] -= offset
        if i[1] > center[1]:
            i[1] += offset
        else:
            i[1] -= offset
    return path


def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its verticies
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in range(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0

def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    """
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = spatial.Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second


def polygons(edges):
    if len(edges) == 0:
        return []

    edges = list(edges.copy())

    shapes = []

    initial = edges[0][0]
    current = edges[0][1]
    points = [initial]
    del edges[0]
    while len(edges) > 0:
        found = False
        for idx, (i, j) in enumerate(edges):
            if i == current:
                points.append(i)
                current = j
                del edges[idx]
                found = True
                break
            if j == current:
                points.append(j)
                current = i
                del edges[idx]
                found = True
                break

        if not found:
            shapes.append(points)
            initial = edges[0][0]
            current = edges[0][1]
            points = [initial]
            del edges[0]

    if len(points) > 1:
        shapes.append(points)

    return shapes


def reduceImageCC(cc: ConnectedComponents, central_text_line: np.ndarray, filter_sigma=5):
    if len(central_text_line) == 0:
        return None

    central_text_line = np.int32(central_text_line[np.lexsort((central_text_line[:, 0],))])
    num_labels, labels, stats, centroids = cc


    intersections = set()

    x_s = np.arange(max(0, central_text_line[0][0]), min(cc.labels.shape[1], central_text_line[-1][0]))
    y_s = np.interp(x_s, central_text_line[:,0], central_text_line[:, 1])
    for x, y in zip(x_s, y_s):
        x, y = int(x), int(y)
        if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0] and labels[y, x] > 0:
            intersections.add(labels[y, x])

    if len(intersections) == 0:
        return None

    min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
    for cp in intersections:
        w = stats[cp, cv2.CC_STAT_WIDTH]
        h = stats[cp, cv2.CC_STAT_HEIGHT]
        a = stats[cp, cv2.CC_STAT_AREA]
        y = stats[cp, cv2.CC_STAT_TOP]
        l = stats[cp, cv2.CC_STAT_LEFT]

        min_x = min(min_x, l)
        min_y = min(min_y, y)
        max_x = max(max_x, l + w)
        max_y = max(max_y, y + h)

    min_x = max(0, min_x - 2)
    max_x = min(labels.shape[1], max_x + 2)
    min_y = max(0, min_y - 2)
    max_y = min(labels.shape[0], max_y + 2)

    cc_image = labels[min_y:max_y, min_x:max_x]
    intersection_image = np.zeros(cc_image.shape, dtype=bool)

    for cp in intersections:
        intersection_image |= cc_image == cp

    intersection_image = intersection_image.astype(np.uint8)
    cv2.polylines(intersection_image, [(central_text_line - (min_x, min_y)).astype(np.int32)], False, (1, ), 8)

    non_intersection_image = (cc_image > 0) ^ intersection_image
    if filter_sigma > 0:
        intersection_image = gaussian_filter(intersection_image.astype(float), sigma=filter_sigma) > 0.1
    intersection_image = intersection_image.astype(bool) ^ (intersection_image.astype(bool) & non_intersection_image)

    return intersection_image, (min_x, min_y)


def extract_components(cc: ConnectedComponents, central_text_line: Coords, staff_lines: List[StaffLine] = None, debug=False) -> List[Coords]:
    if staff_lines is None:
        staff_lines = []

    central_text_line = central_text_line.points
    canvas = (cc.labels > 0) * 255

    result = reduceImageCC(cc, central_text_line, filter_sigma=0 if len(staff_lines) > 0 else 2)
    offset = np.array((0, 0))
    if result is None:
        return []
    else:
        intersection_image, off = result
        offset += off

    if len(staff_lines) > 0:
        intersection_image = intersection_image.astype(np.uint8)
        for sl in staff_lines:
            Coords(sl.coords.points - offset).draw(intersection_image, (0, ), 2)

        cc = ConnectedComponents(*cv2.connectedComponentsWithStats(intersection_image, 4, cv2.CV_32S))
        result = reduceImageCC(cc, central_text_line - offset, filter_sigma=2)

        if result is None:
            return []
        else:
            intersection_image, off = result
            offset += off

    im2, contours, hierarchy = cv2.findContours(intersection_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.reshape((-1, 2)) + offset for c in contours]

    point_list = np.concatenate(tuple(contours), axis=0)
    edges = alpha_shape(point_list, 20)
    polys = polygons(edges)

    approx_fac = 2
    polys = [point_list[poly] for poly in polys]
    polys = [approximate_polygon(p, approx_fac).astype(np.int32) for p in polys]

    if debug:
        canvas = np.stack(((canvas).astype(np.uint8),) * 3, -1)
        cv2.polylines(canvas, [central_text_line.astype(np.int32)], False, [255, 0, 0], thickness=4)
        cv2.polylines(canvas, polys, True, [0, 255, 0])
        cv2.polylines(canvas, contours, True, [0, 0, 255])
        plt.imshow(canvas)
        plt.show()

    return [Coords(p) for p in polys]


if __name__ == '__main__':
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    from main.book import Book, Page
    from omr.datatypes import PcGts
    import os
    import pickle
    book = Book('test')
    page = book.page('Graduel_de_leglise_de_Nevers_536')
    with open(page.file('connected_components_deskewed', create_if_not_existing=True).local_path(), 'rb') as f:
        cc = pickle.load(f)
    line = Coords(np.array([[100, 740], [900, 738]]))
    staff_lines = []
    for mr in PcGts.from_file(page.file('pcgts')).page.music_regions:
        for ml in mr.staffs:
            staff_lines += ml.staff_lines

    extract_components(cc, line, staff_lines, debug=True)
