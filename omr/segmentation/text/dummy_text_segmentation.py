from omr.stafflines.staffline import Staffs
import numpy as np
import json
import cv2


def extract_text(binary: np.ndarray, staffs: Staffs):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - binary, 8, cv2.CV_32S)
    no_text = labels > 0

    central_text_line_y = []
    for i in range(1, len(staffs.staffs)):
        top = staffs.staffs[i - 1].staff_lines[-1].center_y()
        bot = staffs.staffs[i].staff_lines[0].center_y()
        central_text_line_y.append(int((top + bot) / 2))

    central_text_line_y.append(int(staffs.staffs[-1].staff_lines[-1].center_y() + staffs.avg_staff_distance() / 2))

    print(binary.shape, labels.shape)
    print(central_text_line_y)

    intersections = set()

    text = np.zeros(binary.shape, dtype=float)

    for l in central_text_line_y:
        for x in range(0, binary.shape[1]):
            if labels[l, x] > 0:
                intersections.add(labels[l, x])

        cv2.line(binary, (0, l), (binary.shape[1], l), (0.5,), thickness=1)

    print(intersections)


    heights = []
    to_remove = []
    for cp in intersections:
        w = stats[cp, cv2.CC_STAT_WIDTH]
        h = stats[cp, cv2.CC_STAT_HEIGHT]
        a = stats[cp, cv2.CC_STAT_AREA]

        if h / w > 3:
            to_remove.append(cp)
            continue

        text += (labels == cp)
        no_text = no_text ^ (labels == cp)
        heights.append(h)

    for r in to_remove:
        intersections.remove(r)

    avg_text_height = np.median(heights)

    text_line_box_y = []

    for l in central_text_line_y:
        top_offset = -1
        bot_offset = 1

        def cc_sum(offset):
            t = int(l - avg_text_height / 2) + offset
            b = int(t + avg_text_height) + offset
            return np.sum(text[t:b])

        def line_sum(offset):
            return np.sum(text[int(l + offset)]), np.sum(no_text[int(l + offset)])

        top_s = line_sum(top_offset)
        bot_s = line_sum(bot_offset)
        print(top_s, bot_s)
        while top_s[0] > top_s[1]:
            top_offset -= 1
            top_s = line_sum(top_offset)

        while bot_s[0] > bot_s[1]:
            bot_offset += 1
            bot_s = line_sum(bot_offset)

        text_line_box_y.append((top_offset + l, bot_offset + l))

    remaining_intersections = intersections.copy()
    sure_text = np.zeros(text.shape)
    text_blocks = np.zeros(text.shape)
    for l, (t, b) in enumerate(text_line_box_y):
        text[t:b] = 2
        for cp in range(1, num_labels):
            y = stats[cp, cv2.CC_STAT_TOP]
            w = stats[cp, cv2.CC_STAT_WIDTH]
            h = stats[cp, cv2.CC_STAT_HEIGHT]
            a = stats[cp, cv2.CC_STAT_AREA]
            c = (t + b) / 2
            if y > c + avg_text_height * 0.25 or y + h < c - avg_text_height * 0.25:
                continue

            if y >= t and y + h <= b:
                if cp in intersections:
                    remaining_intersections.remove(cp)
                f = 5 if cp in intersections else 1
                sure_text += (labels == cp) * f
                text_blocks += (labels == cp) * (l + 1)
                continue

            te = t - avg_text_height * 0.5
            be = b + avg_text_height * 0.5
            c = (te + be) // 2
            if y > c + avg_text_height * 0.25 or y + h < c - avg_text_height * 0.25:
                continue
            if be >= y + h > c - avg_text_height * 0.25 and c + avg_text_height > y >= te and 0.5 * avg_text_height < h < 2 * avg_text_height:
                if cp in intersections:
                    remaining_intersections.remove(cp)
                sure_text += (labels == cp) * 2
                text_blocks += (labels == cp) * (l + 1)
                continue

    for l, (t, b) in enumerate(text_line_box_y):
        for cp in remaining_intersections:
            c = (labels == cp).astype(np.uint8)
            c[t:b] += 1
            sure_text += c == 2
            text_blocks += (c == 2) * (l + 1)

    lines = []
    for l, _ in enumerate(text_line_box_y):
        mask = text_blocks == (l + 1)
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        lines.append(mask[x0:x1, y0:y1])


    print(avg_text_height)

    return sure_text, text, no_text, text_blocks, lines


if __name__ == '__main__':
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    binary = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'deskewed_binary.png'))
    gray = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'deskewed_gray.jpg'))
    with open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000001', 'detected_staffs.json'), 'r') as f:
        staffs_json = json.load(f)
    staffs = Staffs.from_json(staffs_json)
    sure_text, text, no_text, text_blocks, lines = extract_text(np.array(binary), staffs)
    f, ax = plt.subplots(len(lines), 1)
    for l, a in zip(lines, ax):
        a.imshow(l)

    plt.show()
    f, ax = plt.subplots(2, 4, True, True)
    ax[0, 0].imshow(sure_text)
    ax[0, 1].imshow(text)
    ax[0, 2].imshow(no_text)
    ax[0, 3].imshow(binary)
    ax[1, 0].imshow(text_blocks)
    plt.show()

