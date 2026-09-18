"""Centre-distance localisation evaluation (PCGTS has no ground-truth boxes)."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from omr.discovery.schema import Box


@dataclass
class Matching:
    candidate_to_gt: Dict[str, str]
    gt_to_candidate: Dict[str, str]
    normalized_distances: Dict[str, float]


StaffLineKey = Union[str, Tuple[str, str]]


def _staff_space(staff_space_by_line: Dict[StaffLineKey, float], page: str, line_id: str) -> float:
    return staff_space_by_line.get((page, line_id), staff_space_by_line.get(line_id, 1e-3))

def match(candidates, gt, staff_space_by_line: Dict[StaffLineKey, float], radius: float) -> Matching:
    c2g, g2c, distances = {}, {}, {}
    by_line = sorted({(g.page, g.line_id) for g in gt} | {(c.page, c.line_id) for c in candidates})
    for page, line_id in by_line:
        cs = [c for c in candidates if c.page == page and c.line_id == line_id]
        gs = [g for g in gt if g.page == page and g.line_id == line_id]
        if not cs or not gs: continue
        ss = max(_staff_space(staff_space_by_line, page, line_id), 1e-12)
        cost = np.array([[np.hypot(c.center_x-g.x, c.center_y-g.y)/ss for g in gs] for c in cs])
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            d = float(cost[row, col])
            if d <= radius:
                c2g[cs[row].id] = gs[col].id; g2c[gs[col].id] = cs[row].id
                distances[cs[row].id] = d
    return Matching(c2g, g2c, distances)


def localization_metrics(candidates, gt, staff_space_by_line, cfg):
    matching = match(candidates, gt, staff_space_by_line, cfg.match_radius_staff_space)
    tp, n_c, n_g = len(matching.candidate_to_gt), len(candidates), len(gt)
    precision = tp/n_c if n_c else 0.0; recall = tp/n_g if n_g else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    gt_by_id = {g.id:g for g in gt}
    missed = {}
    for g in gt:
        if g.id not in matching.gt_to_candidate: missed[g.family] = missed.get(g.family, 0)+1
    duplicates = splits = 0
    unmatched=[c for c in candidates if c.id not in matching.candidate_to_gt]
    for g in gt:
        ss = max(_staff_space(staff_space_by_line, g.page, g.line_id), 1e-12)
        extras = sum(c.page == g.page and c.line_id == g.line_id and
                     np.hypot(c.center_x-g.x,c.center_y-g.y)/ss <= cfg.match_radius_staff_space
                     for c in unmatched)
        duplicates += extras
        splits += extras > 0 and g.id in matching.gt_to_candidate
    merges = sum(sum(g.page==c.page and g.line_id==c.line_id and c.box.contains_point(g.x,g.y)
                     for g in gt) >= 2 for c in candidates)
    ious = []
    for cid, gid in matching.candidate_to_gt.items():
        c = next(c for c in candidates if c.id == cid); g=gt_by_id[gid]
        ss=max(_staff_space(staff_space_by_line,g.page,g.line_id),1e-12); side=cfg.derived_gt_box_staff_space*ss
        ious.append(c.box.iou(Box(g.x-side/2,g.y-side/2,side,side)))
    ds=list(matching.normalized_distances.values())
    return ({'n_candidates':n_c,'n_groundtruth':n_g,'n_matched':tp,'precision':precision,
             'recall':recall,'f1':f1,'mean_norm_center_distance':float(np.mean(ds)) if ds else None,
             'median_norm_center_distance':float(np.median(ds)) if ds else None,
             'n_duplicates':int(duplicates),'n_merges':int(merges),'n_splits':int(splits),
             'missed_by_family':missed,
             'iou_mean_derived_gt_box':float(np.mean(ious)) if ious else None}, matching)
