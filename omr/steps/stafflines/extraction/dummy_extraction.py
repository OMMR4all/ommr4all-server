from PIL import Image
import json
import numpy as np
from omr.stafflines.staffline import Staffs, StaffLine
import cv2


def extract_staffs(binary, staffs):
    binary = np.array(binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - binary, 8, cv2.CV_32S)
    no_text = labels > 0

    music_p = np.zeros(binary.shape, dtype=np.int32)
    music_l = np.zeros(binary.shape, dtype=np.int32)

    for staff_id, staff in enumerate(staffs.staffs, start=1):
        if len(staff.staff_lines) == 0:
            continue

        top_line: StaffLine = staff.staff_lines[0]
        bot_line: StaffLine = staff.staff_lines[-1]

        top_border = top_line.dewarped_y() - staff.avg_staff_line_distance
        bot_border = bot_line.dewarped_y() + staff.avg_staff_line_distance
        t = int(top_border)
        b = int(bot_border)
        line = labels[t:b,:] > 0
        music_l[t:b] += line * staff_id


    return music_p, music_l


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ommr4all.settings import PRIVATE_MEDIA_ROOT
    import os
    binary = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'dewarped_binary.png'))
    gray = Image.open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'dewarped_gray.jpg'))
    with open(os.path.join(PRIVATE_MEDIA_ROOT, 'demo', 'page00000002', 'annotation.json'), 'r') as f:
        staffs_json = json.load(f)
    staffs = Staffs.from_json(staffs_json)
    music_p, music_l = extract_staffs(binary, staffs)

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(binary)
    ax[1].imshow(music_p)
    ax[2].imshow(music_l)

    plt.show()
