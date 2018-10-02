from omr.stafflines.staffline import Staffs
from omr.preprocessing.util.connected_compontents import ConnectedComponents
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from scipy import signal, spatial
from skimage.measure import approximate_polygon
from omr.stafflines.text_line import TextBoundaries, TextLine

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


def extract_text(cc: ConnectedComponents, central_text_line: np.ndarray, debug=False) -> TextLine:
    central_text_line = np.int32(central_text_line[np.lexsort((central_text_line[:, 0],))])
    num_labels, labels, stats, centroids = cc
    no_text = labels > 0


    canvas = no_text.astype(np.uint8) * 255
    print(canvas.shape, central_text_line)

    intersections = set()

    text = np.zeros(labels.shape, dtype=float)
    cv2.polylines(canvas, [central_text_line], False, 128, thickness=1)

    x_s = np.arange(central_text_line[0][0], central_text_line[-1][0])
    y_s = np.interp(x_s, central_text_line[:,0], central_text_line[:, 1])
    for x, y in zip(x_s, y_s):
        x, y = int(x), int(y)
        if labels[y, x] > 0:
            intersections.add(labels[y, x])

    if len(intersections) == 0:
        return TextLine()

    print(intersections)

    heights = []
    to_remove = []
    center = []
    for cp in intersections:
        w = stats[cp, cv2.CC_STAT_WIDTH]
        h = stats[cp, cv2.CC_STAT_HEIGHT]
        a = stats[cp, cv2.CC_STAT_AREA]
        y = stats[cp, cv2.CC_STAT_TOP]
        l = stats[cp, cv2.CC_STAT_LEFT]

        if h / w > 10 or h == 0 or w == 0:
            to_remove.append(cp)
            continue

        sub = labels[y:y+h, l:l+w] == cp
        for x in range(w):
            ssub = np.nonzero(sub[:, x])[0]
            heights.append(ssub[-1] - ssub[0])
            center.append((int(x + l), int(y + (ssub[0] + ssub[-1]) / 2)))

        text += (labels == cp)
        no_text = no_text ^ (labels == cp)

    for r in to_remove:
        intersections.remove(r)

    center = np.array(sorted(center, key=lambda p: p[0]))
    print(center, len(center), center.shape)
    center_x = np.arange(center[0, 0], center[-1, 0])
    center = np.interp(center_x, center[:, 0], center[:, 1])
    center = center.astype(np.int32)

    avg_text_height = int(np.median(heights))
    print("Medium Height: {}".format(avg_text_height))
    avg = 2
    indices_x = np.arange(0, len(center) + 0.1, avg_text_height * avg).astype(int)
    center_line = np.array([np.median(center[max(0, int(x - avg_text_height * avg)): min(len(center), int(x + avg_text_height * avg))]) for x in indices_x], dtype=np.int32)
    center_x = np.linspace(center_x[0], center_x[-1], len(center_line)).astype(int)

    heights = [h for h in heights if h > avg_text_height * 1.2 and h <= 3 * avg_text_height]
    avg_cap_text_height = np.median(heights)
    print("Medium 2nd Height: {}".format(np.median(heights)))

    top_line = center_line - avg_text_height / 2
    bot_line = center_line + avg_text_height / 2
    cap_top_line = bot_line - avg_cap_text_height
    cap_bot_line = top_line + avg_cap_text_height

    all_points = np.zeros((0, 2))

    for cp in intersections:
        w = stats[cp, cv2.CC_STAT_WIDTH]
        h = stats[cp, cv2.CC_STAT_HEIGHT]
        a = stats[cp, cv2.CC_STAT_AREA]
        y = stats[cp, cv2.CC_STAT_TOP]
        l = stats[cp, cv2.CC_STAT_LEFT]

        img = np.array(labels[y:y+h, l:l+w] == cp)
        top = (np.interp(np.arange(l, l+w + 0.1), center_x, top_line) - y).astype(int)
        bot = top + 2 * (avg_cap_text_height - avg_text_height) + avg_text_height
        bot = np.minimum(bot, h).astype(int)
        top = np.maximum(top, 0).astype(int)
        for i, x in enumerate(range(w)):
            img[top[i]:bot[i],x] = 0

        n, cl, s, c = cv2.connectedComponentsWithStats(img.astype(np.uint8))
        to_rem = np.full(img.shape, False)
        for sub_c in range(1, n):
            s_w = s[sub_c, cv2.CC_STAT_WIDTH]
            s_h = s[sub_c, cv2.CC_STAT_HEIGHT]
            s_a = s[sub_c, cv2.CC_STAT_AREA]

            if s_w < avg_text_height * 2 and s_h < avg_text_height:
                pass
            else:
                to_rem = to_rem | (cl == sub_c)

        out = ((labels[y:y+h, l:l+w] == cp) & (1 - to_rem))
        filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        conv = signal.convolve2d(out, filter, mode='same')
        edges = out & (1 - (conv == 5))
        points = np.column_stack(np.where(edges))

        points[:, 0] += y
        points[:, 1] += l

        all_points = np.append(all_points, points, axis=0)

    edges = alpha_shape(all_points, avg_text_height / 2)
    polys = polygons(edges)

    polys = [np.flip(all_points[poly], axis=1) for poly in polys]
    polys = [scale_polygon(approximate_polygon(p, avg_text_height / 5), avg_text_height / 5) for p in polys]

    text_line = TextLine(polys, TextBoundaries(np.column_stack((center_x, cap_top_line)),
                                               np.column_stack((center_x, top_line)),
                                               np.column_stack((center_x, bot_line)),
                                               np.column_stack((center_x, cap_bot_line))))

    if debug:
        canvas = np.stack(((canvas).astype(np.uint8),) * 3, -1)
        text_line.draw(canvas)
        plt.imshow(canvas)
        plt.show()

    return text_line


if __name__ == '__main__':
    from gregorian_annotator_server.settings import PRIVATE_MEDIA_ROOT
    import os
    import pickle
    with open(os.path.join(PRIVATE_MEDIA_ROOT, 'test', 'Graduel_de_leglise_de_Nevers_536', 'connected_components_deskewed.pkl'), 'rb') as f:
        cc = pickle.load(f)
    line = np.array([[100, 383], [900, 380]])
    extract_text(cc, line, debug=True)
