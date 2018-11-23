import numpy as np
import cv2
from tqdm import tqdm
import logging
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.signal import convolve2d
from main.book import Book, Page, File
from omr.datatypes import *
from typing import List
import matplotlib.pyplot as plt
from omr.preprocessing.binarizer.ocropus_binarizer import binarize
from scipy.spatial import ConvexHull

logger = logging.getLogger("Staffline Detector")


def imshow(img):
    plt.imshow(img)
    plt.show()


def detect(binary: np.ndarray, gray: np.ndarray, debug=False) -> MusicLines:
    #filtered = gaussian_filter1d(img[:,:,2] + img[:,:,1] - img[:,:,0], 3, axis=1)
    gray = cv2.bilateralFilter((gray * 255).astype(np.uint8), 5, 75, 75)
    gray = (1 - np.clip(convolve2d(1 - gray, np.full((1, 10), 0.2)), 0, 255)) / 255
    binary = binarize(gray)

    binarized = 1 - binary
    morph = binary_erosion(binarized, structure=np.full((5, 1), 1))
    morph = binary_dilation(morph, structure=np.full((5, 1), 1))

    staffs = (binarized ^ morph)

    if debug:
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(binary)
        ax[1].imshow(morph)
        ax[2].imshow(staffs)
        plt.show()

    morph = binary_dilation(morph, structure=np.full((3, 3), 1), iterations=2)
    binary_text = binarized & morph


    staffs = staffs.astype(np.uint8)

    staffs, point_list, line_distance = detect_staffs(staffs)


    print(staffs)
    staffs = [staff for staff in staffs if len(staff) >= 3 and len(staff) <= 6]
    print(staffs)
    staffs = [[point_list[l_id] for l_id in staff] for staff in staffs]
    #staffs_out = Staffs([Staff([StaffLine(line) for line in staff]) for staff in staffs])
    #return staffs_out

    def normalize_staff(staff):
        if len(staff) > 4:
            ls = [line[-1][0] - line[0][0] for line in staff]
            med_l = np.median(ls)
            to_remove = [i for i in range(len(staff)) if ls[i] > 2 * med_l or ls[i] < 0.5 * med_l]
            for r in reversed(to_remove):
                del staff[r]

            if len(staff) < 2:
                return None

        sxs = sorted([line[0][0] for line in staff])
        exs = sorted([line[-1][0] for line in staff])

        def best_dist(sxs, min_max):
            ds = []
            for i, s1 in enumerate(sxs):
                for j, s2 in enumerate(sxs[i+1:]):
                    ds.append((np.abs(s1 - s2), i, j + i + 1))

            ds = sorted(ds, key=lambda s: s[0])
            return min_max([sxs[ds[0][i + 1]] for i in range(2)])

        sx = best_dist(sxs, min)
        ex = best_dist(exs, max)

        for line in staff:
            x, y = zip(*line)
            if x[0] < sx:
                sy = np.interp(sx, x, y)
                while len(line) > 0 and line[0][0] <= sx:
                    del line[0]

                line.insert(0, (sx, sy))

            elif x[0] > sx:
                p1 = line[0]
                p2 = (p1[0] + line_distance, np.interp(p1[0] + line_distance, x, y))

                sy = np.interp(sx, [p1[0], p2[0]], [p1[1], p2[1]])
                line.insert(0, (sx, sy))


            if x[-1] > ex:
                sy = np.interp(ex, x, y)
                while line[-1][0] >= ex:
                    del line[-1]

                line.append((ex, sy))

            elif x[-1] < ex:
                p1 = line[-1]
                p2 = (p1[0] - line_distance, np.interp(p1[0] - line_distance, x, y))

                sy = np.interp(ex, [p1[0], p2[0]], [p1[1], p2[1]])
                line.append((ex, sy))

        def full_points(line):
            x, y = zip(*line)
            y = np.interp(range(sx, ex), x, y)
            x = range(sx, ex)
            return np.array(list(zip(x, y)))

        staff = np.asarray(list(map(full_points, staff)))

        return np.array(sorted(staff, key=lambda line: line[0][1])), sx, ex

    logger.debug(len(staffs))
    staffs = [l for l in map(normalize_staff, staffs) if l]
    logger.debug(len(staffs))
    logger.debug([len(l[0]) for l in staffs])

    def to_staff_line(line) -> StaffLine:
        coords = Coords(line)
        coords.approximate(line_distance / 10)
        return StaffLine(coords)

    def to_staff(staff) -> MusicLine:
        global lines_only
        staff = staff[0]
        lines: List[StaffLine] = list(map(to_staff_line, staff))
        all_points = np.concatenate(tuple([f.coords.points for f in lines]), axis=0)
        coords = Coords(all_points[ConvexHull(all_points).vertices])
        return MusicLine(coords=coords, staff_lines=StaffLines(lines))

    staffs = StaffLines(map(to_staff, staffs))

    return MusicLines(staffs)


def detect_staffs(staff_binary: np.ndarray):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(staff_binary, 8, cv2.CV_32S)

    lines = []
    for l in tqdm(range(1, num_labels), total=num_labels-1):
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        a = stats[l, cv2.CC_STAT_AREA]
        if h <= 2 and w >= 20:
            lines.append(l)
        elif h / w > 0.5 or w < 30 or a < 20:
            labels *= labels != l
        else:
            lines.append(l)


    # detect staff line distance
    distances = []
    for i, l1 in enumerate(lines):
        y1 = stats[l1, cv2.CC_STAT_TOP]
        h1 = stats[l1, cv2.CC_STAT_HEIGHT]
        c1 = y1 + h1 / 2
        for l2 in lines[i + 1:]:
            y2 = stats[l2, cv2.CC_STAT_TOP]
            h2 = stats[l2, cv2.CC_STAT_HEIGHT]
            c2 = y2 + h2 / 2
            d = np.abs(c2 - c1)
            if d < 100 and d > 5:
                distances += [d]

    uniques, counts = np.unique(distances, return_counts=True)
    line_distance = uniques[np.argmax(counts)]

    print("Line distance {}".format(line_distance))

    def to_point_list(stats, label):
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        a = stats[label, cv2.CC_STAT_AREA]
        y = stats[label, cv2.CC_STAT_TOP]
        x = stats[label, cv2.CC_STAT_LEFT]
        points = []
        for px in range(x,x+w):
            sy = y
            ey = min(y + h, labels.shape[0] - 1)
            while labels[sy, px] != label and sy + 1 < ey:
                sy += 1

            while labels[ey, px] != label and ey > y:
                ey -= 1

            points.append((px, (ey + sy) / 2.0))

        return points

    def sort_line(line):
        return sorted(line, key=lambda p: p[0])

    point_list = [sort_line(to_point_list(stats, label)) for label in lines]


    for max_dist in [10, 50, 100, 200, 500]:
        print(len(point_list))
        i = 0
        while i < len(point_list):
            l1 = point_list[i]
            x1b, y1b = l1[0]
            x1e, y1e = l1[-1]

            found = False
            for i2 in range(i + 1,len(point_list)):
                l2 = point_list[i2]
                x2b, y2b = l2[0]
                x2e, y2e = l2[-1]
                if x1e < x2b and x2b - x1e < max_dist:
                    if np.abs(y1e - y2b) < line_distance / 3:
                        point_list[i] = sort_line(l1 + l2)
                        del point_list[i2]
                        found = True
                        break
                elif x2e < x1b and x1b - x2e < max_dist:
                    if np.abs(y1b - y2e) < line_distance / 3:
                        point_list[i] = sort_line(l2 + l1)
                        del point_list[i2]
                        found = True
                        break

            if not found:
                i += 1

    def p2i(p):
        return int(p[0]), int(p[1])


    point_list = [l for l in point_list if l[-1][0] - l[0][0] > 10]

    point_list_y_center = [np.mean([y for x, y in l]) for l in point_list]

    staffs = []

    for i, center_y in enumerate(point_list_y_center):
        found = None

        for staff in staffs:
            for s_i, s in enumerate(staff):
                o_y = point_list_y_center[s]
                # allow even to skip a line
                if np.abs(o_y - center_y) <2.1 * line_distance:
                    found = staff
                    break

            if found:
                break

        if not found:
            staffs.append([i])
        else:
            found.append(i)

    return staffs, point_list, line_distance


if __name__=='__main__':
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import os
    from PIL import Image
    binary = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'binary_deskewed.png'))
    gray = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'gray_deskewed.jpg'))
    staffs = detect(np.array(binary) // 255, np.array(gray) / 255)
    img = np.array(Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'color_deskewed.jpg')), dtype=np.uint8)
    staffs.draw(img)
    plt.imshow(img)
    plt.show()
