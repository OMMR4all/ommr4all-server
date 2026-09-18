"""OpenCV visualisations of candidates, clusters and neumes."""
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np

from database.file_formats.pcgts import PageScaleReference, Point
from omr.discovery.regions import page_box_to_crop
from omr.discovery.schema import GroupingState, ReviewState


def _mkdir(path): os.makedirs(os.path.dirname(path), exist_ok=True)


def _image(page):
    import cv2
    path = page.file('color_highres_preproc', create_if_not_existing=True).local_path()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    return image


def _image_box(page_obj, box):
    p0 = page_obj.page_to_image_scale(Point(box.x, box.y), PageScaleReference.HIGHRES)
    p1 = page_obj.page_to_image_scale(Point(box.right(), box.bottom()), PageScaleReference.HIGHRES)
    return tuple(np.round([p0.x, p0.y, p1.x, p1.y]).astype(int))


def page_overlay(page, candidates, out_path: str, show_rejected: bool = True) -> None:
    import cv2
    image = _image(page)
    colors = {ReviewState.UNREVIEWED: (0, 220, 255), ReviewState.ACCEPTED: (0, 190, 0),
              ReviewState.MODIFIED: (255, 120, 0), ReviewState.REJECTED: (0, 0, 230)}
    page_obj = page.pcgts().page
    for c in candidates:
        if c.review_state == ReviewState.REJECTED and not show_rejected: continue
        x0, y0, x1, y1 = _image_box(page_obj, c.box)
        color = colors[c.review_state]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image, c.cluster_id, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1, cv2.LINE_AA)
    _mkdir(out_path)
    if not cv2.imwrite(out_path, image): raise IOError('could not write ' + out_path)


def _tile(items, header: str, out_path: str, max_items: int = 144):
    import cv2
    thumb_h, cell_w, caption_h, header_h = 64, 130, 22, 40
    items = items[:max_items]
    cols = min(8, max(1, len(items)))
    rows = max(1, int(np.ceil(len(items) / cols)))
    canvas = np.full((header_h + rows * (thumb_h + caption_h), cols * cell_w, 3), 245, np.uint8)
    cv2.putText(canvas, header[:max(20, cols * 18)], (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)
    for i, (img, caption) in enumerate(items):
        r, c = divmod(i, cols)
        if img is None or img.size == 0: continue
        scale = min(cell_w / img.shape[1], thumb_h / img.shape[0])
        resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                   max(1, int(img.shape[0] * scale))))
        y = header_h + r * (thumb_h + caption_h)
        x = c * cell_w + (cell_w - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(canvas, caption[:20], (c * cell_w + 3, y + thumb_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
    _mkdir(out_path)
    if not cv2.imwrite(out_path, canvas): raise IOError('could not write ' + out_path)


def _crop_member(crop, box):
    x, y, w, h = page_box_to_crop(crop, box)
    pad = 0.3 * crop.staff_space_px
    x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
    x1, y1 = min(crop.image.shape[1], int(np.ceil(x + w + pad))), \
             min(crop.image.shape[0], int(np.ceil(y + h + pad)))
    return crop.image[y0:y1, x0:x1][:, :, ::-1]


def _crop_for(crops, page, line_id):
    return crops.get((page, line_id), crops.get(line_id))


def safe_cluster_id(cluster_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', cluster_id)


def cluster_contact_sheet(store, cluster_id: str, crops: Dict, out_path: str,
                          max_items: int = 144) -> None:
    candidates = [c for c in store.ordered_candidates() if c.cluster_id == cluster_id]
    histogram = Counter(c.review_state.value + '/' + c.label.family.value for c in candidates)
    header = 'cluster {} | n={} | {}'.format(cluster_id, len(candidates),
                                             ', '.join('{}={}'.format(k, v) for k, v in histogram.items()))
    items = []
    for i, c in enumerate(candidates):
        crop = _crop_for(crops, c.page, c.line_id)
        if crop is not None: items.append((_crop_member(crop, c.box), '{} {}'.format(i, c.page)))
    _tile(items, header, out_path, max_items)


def neume_page_overlay(page, store, out_path: str) -> None:
    import cv2
    image = _image(page)
    page_obj = page.pcgts().page
    for n in store.neumes_of_page(page.page):
        x0, y0, x1, y1 = _image_box(page_obj, n.box)
        color = (255, 120, 0) if n.grouping_state == GroupingState.MODIFIED else (180, 0, 180)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        components = [store.candidates[cid] for cid in n.component_ids]
        for relation, a, b in zip(n.relations, components, components[1:]):
            pa = page_obj.page_to_image_scale(Point(a.center_x, a.center_y), PageScaleReference.HIGHRES)
            pb = page_obj.page_to_image_scale(Point(b.center_x, b.center_y), PageScaleReference.HIGHRES)
            p0, p1 = (int(pa.x), int(pa.y)), (int(pb.x), int(pb.y))
            if relation.kind == 'looped': cv2.line(image, p0, p1, color, 3)
            else:
                for t in np.arange(0, 1, 0.2):
                    q0 = tuple(np.round((1-t)*np.array(p0)+t*np.array(p1)).astype(int))
                    q1 = tuple(np.round((1-min(1,t+0.1))*np.array(p0)+min(1,t+0.1)*np.array(p1)).astype(int))
                    cv2.line(image, q0, q1, color, 2)
        cv2.putText(image, n.grouping_state.value + '/' + n.neume_type,
                    (x0, max(10, y0-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    _mkdir(out_path)
    if not cv2.imwrite(out_path, image): raise IOError('could not write ' + out_path)


def neume_contact_sheet(store, cluster_id: str, crops: Dict, out_path: str,
                        max_items: int = 144) -> None:
    neumes = [n for n in store.ordered_neumes() if n.cluster_id == cluster_id]
    histogram = Counter(n.grouping_state.value + '/' + n.neume_type for n in neumes)
    header = 'neume cluster {} | n={} | {}'.format(cluster_id, len(neumes),
                                                   ', '.join('{}={}'.format(k,v) for k,v in histogram.items()))
    items = []
    for i, n in enumerate(neumes):
        crop = _crop_for(crops, n.page, n.line_id)
        if crop is not None: items.append((_crop_member(crop, n.box), '{} {}'.format(i, n.page)))
    _tile(items, header, out_path, max_items)
