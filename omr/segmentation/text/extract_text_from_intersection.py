from omr.stafflines.staffline import Staffs
from omr.preprocessing.util.connected_compontents import ConnectedComponents
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt


def extract_text(cc: ConnectedComponents, central_text_line: np.ndarray):
    central_text_line = np.int32(central_text_line)
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

        if h / w > 3 or h == 0 or w == 0:
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

    points = np.column_stack((center_x, top_line)).astype(int)
    cv2.polylines(canvas, [points], False, 160, thickness=1)
    points = np.column_stack((center_x, cap_top_line)).astype(int)
    cv2.polylines(canvas, [points], False, 80, thickness=1)
    points = np.column_stack((center_x, bot_line)).astype(int)
    cv2.polylines(canvas, [points], False, 160, thickness=1)
    points = np.column_stack((center_x, cap_bot_line)).astype(int)
    cv2.polylines(canvas, [points], False, 80, thickness=1)
    plt.imshow(canvas)
    plt.show()

    for cp in intersections:
        w = stats[cp, cv2.CC_STAT_WIDTH]
        h = stats[cp, cv2.CC_STAT_HEIGHT]
        a = stats[cp, cv2.CC_STAT_AREA]
        y = stats[cp, cv2.CC_STAT_TOP]
        l = stats[cp, cv2.CC_STAT_LEFT]

        img = labels[y:y+h, l:l+w] == cp
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

            if s_w < avg_text_height and s_h < avg_text_height:
                pass
            else:
                to_rem = to_rem | (cl == sub_c)

        out = (labels[y:y+h, l:l+w] == cp) & (1 - to_rem)








if __name__ == '__main__':
    from gregorian_annotator_server.settings import PRIVATE_MEDIA_ROOT
    import os
    import pickle
    with open(os.path.join(PRIVATE_MEDIA_ROOT, 'test', 'Graduel_de_leglise_de_Nevers_536', 'connected_components_deskewed.pkl'), 'rb') as f:
        cc = pickle.load(f)
    line = np.array([[100, 383], [900, 380]])
    extract_text(cc, line)
