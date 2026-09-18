"""Class-agnostic symbol localisation: interface and the shared scale prior.

Nothing in this module (or in any discovery method) assigns a semantic class. A method
returns geometry plus a confidence; interpretation happens later, by a human, in
`omr.discovery.review`.

Every threshold is expressed in staff spaces and converted to pixels here, so the same
configuration works on books of different resolutions.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from omr.discovery.config import DiscoveryConfig
from omr.discovery.features.base import SpatialFeatureMap

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """One candidate region in crop pixels."""
    x: float
    y: float
    w: float
    h: float
    score: float
    method: str
    #: optional bool mask of exactly (round(h), round(w)) pixels
    mask: Optional[np.ndarray] = None

    def iou(self, other: 'Proposal') -> float:
        dx = min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        dy = min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0.0
        inter = dx * dy
        return inter / (self.w * self.h + other.w * other.h - inter)


@dataclass
class DiscoveryContext:
    """Everything a discovery method may look at for one staff line."""
    crop: 'object'  # omr.discovery.regions.StaffCrop
    cfg: DiscoveryConfig
    #: ink mask with the staff lines removed, same shape as `crop.image`
    ink: np.ndarray
    feature_map: Optional[SpatialFeatureMap] = None


class SymbolDiscoveryMethod(ABC):
    name: str = 'base'

    @abstractmethod
    def discover(self, ctx: DiscoveryContext) -> List[Proposal]:
        ...


def _vertical_run_length(mask: np.ndarray) -> np.ndarray:
    """For every True pixel the length of the maximal vertical run it belongs to."""
    h = mask.shape[0]
    up = np.zeros(mask.shape, dtype=np.int32)
    down = np.zeros(mask.shape, dtype=np.int32)
    up[0] = mask[0]
    for y in range(1, h):
        up[y] = np.where(mask[y], up[y - 1] + 1, 0)
    down[h - 1] = mask[h - 1]
    for y in range(h - 2, -1, -1):
        down[y] = np.where(mask[y], down[y + 1] + 1, 0)
    run = up + down - 1
    return np.where(mask, run, 0)


def _horizontal_run_length(mask: np.ndarray) -> np.ndarray:
    """For every True pixel the length of the maximal horizontal run it belongs to."""
    return _vertical_run_length(mask.T).T


def rasterize_staff_lines(shape: Tuple[int, int], staff_lines_px: List[np.ndarray],
                          thickness: int) -> np.ndarray:
    """A mask of the annotated staff line strokes, thickened by `thickness` pixels."""
    import cv2
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    for points in staff_lines_px:
        if points.shape[0] < 2:
            continue
        polyline = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polyline], isClosed=False, color=1, thickness=max(1, thickness))
    return canvas > 0


def remove_staff_lines(binary: np.ndarray, staff_lines_px: List[np.ndarray],
                       staff_space_px: float, cfg: DiscoveryConfig) -> np.ndarray:
    """Delete the staff lines from an ink mask without eating the symbols.

    A staff line is *thin* and *long*: ink is removed only where its vertical run is at most
    `staff_line_removal_max_run_staff_space` staff spaces and it either continues horizontally
    for `staff_line_removal_min_length_staff_space` staff spaces or lies on an annotated staff
    line. A note body has a long vertical run and therefore survives even where a line crosses
    it, and the annotated geometry is only a hint -- the ink of a line that the annotation
    misses by a few pixels is still removed by the length rule.
    """
    import cv2
    max_run = max(1.0, staff_space_px * cfg.staff_line_removal_max_run_staff_space)
    min_length = max(2.0, staff_space_px * cfg.staff_line_removal_min_length_staff_space)
    # the annotated polyline is thickened to cover the whole stroke, not just its centre
    zone = rasterize_staff_lines(binary.shape, staff_lines_px, int(round(2 * max_run)))
    vertical = _vertical_run_length(binary)
    horizontal = _horizontal_run_length(binary)
    thin = (vertical > 0) & (vertical <= max_run)
    out = binary & ~(thin & ((horizontal >= min_length) | zone))

    close_height = int(round(staff_space_px * cfg.close_kernel_staff_space))
    if close_height >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_height))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return out


def scale_gate(proposals: List[Proposal], staff_space_px: float,
               cfg: DiscoveryConfig) -> List[Proposal]:
    """Drop proposals that cannot be a single symbol by size.

    Only a lower bound on the area and an upper bound on the height are enforced. A box that
    is too *wide* is deliberately kept: it usually covers two touching symbols, and keeping it
    costs one false positive while dropping it costs two misses. The evaluation counts those
    boxes explicitly as `n_merges`, which is what the failure mode report needs.
    """
    min_area = (staff_space_px ** 2) * cfg.min_area_staff_space_sq
    max_height = staff_space_px * cfg.max_height_staff_space
    out = []
    for p in proposals:
        if p.w <= 0 or p.h <= 0:
            continue
        if p.w * p.h < min_area:
            continue
        if p.h > max_height:
            continue
        out.append(p)
    return out


def nms(proposals: List[Proposal], iou_threshold: float) -> List[Proposal]:
    """Greedy non-maximum suppression, highest score first."""
    kept: List[Proposal] = []
    for p in sorted(proposals, key=lambda q: -q.score):
        if all(p.iou(k) <= iou_threshold for k in kept):
            kept.append(p)
    return kept


def split_wide(mask: np.ndarray, staff_space_px: float,
               cfg: DiscoveryConfig) -> List[Tuple[np.ndarray, int]]:
    """Cut a too wide component at the minima of its column ink profile.

    Returns `(sub_mask, x_offset)` pairs; a component that is narrow enough is returned as is.
    """
    from scipy.signal import find_peaks
    max_width = staff_space_px * cfg.max_width_staff_space
    if mask.shape[1] <= max_width:
        return [(mask, 0)]

    profile = mask.sum(axis=0).astype(float)
    distance = max(1, int(round(staff_space_px * cfg.split_min_distance_staff_space)))
    minima, _ = find_peaks(-profile, distance=distance)
    cuts = [int(c) for c in minima if 0 < c < mask.shape[1] - 1]
    if not cuts:
        return [(mask, 0)]

    out = []
    bounds = [0] + cuts + [mask.shape[1]]
    for left, right in zip(bounds, bounds[1:]):
        if right - left <= 0:
            continue
        piece = mask[:, left:right]
        if not piece.any():
            continue
        out.append((piece, left))
    return out or [(mask, 0)]


def split_stacked(mask: np.ndarray, staff_space_px: float,
                  cfg: DiscoveryConfig) -> List[Tuple[np.ndarray, int]]:
    """Split vertically stacked noteheads joined into one connected component.

    Strong, vertically separated distance-transform maxima identify compact ink lobes. A
    horizontal cut halfway between adjacent lobes preserves two pitch-bearing instances even
    when residual staff ink or a short connector joins them. Conservative staff-relative
    height, width and peak-radius gates leave ordinary tall single symbols untouched.

    Returns ``(sub_mask, y_offset)`` pairs; an ambiguous component is returned unchanged.
    """
    import cv2

    min_height = staff_space_px * cfg.stacked_split_min_height_staff_space
    max_width = staff_space_px * cfg.stacked_split_max_width_staff_space
    if mask.shape[0] < min_height or mask.shape[1] > max_width:
        return [(mask, 0)]

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = staff_space_px * cfg.stacked_split_min_peak_radius_staff_space
    neighborhood = max(3, int(round(
        staff_space_px * cfg.stacked_split_min_peak_dy_staff_space)))
    if neighborhood % 2 == 0:
        neighborhood += 1
    local_max = distance >= cv2.dilate(
        distance, np.ones((neighborhood, neighborhood), dtype=np.uint8))
    local_max &= distance >= radius
    n_labels, labels = cv2.connectedComponents(local_max.astype(np.uint8), connectivity=8)

    peaks = []
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        best = int(np.argmax(distance[ys, xs]))
        peaks.append((float(xs[best]), float(ys[best])))

    min_dy = staff_space_px * cfg.stacked_split_min_peak_dy_staff_space
    max_dy = staff_space_px * cfg.stacked_split_max_peak_dy_staff_space
    max_dx = staff_space_px * cfg.stacked_split_max_peak_dx_staff_space
    participating = set()
    for i, left in enumerate(peaks):
        for j, right in enumerate(peaks[i + 1:], i + 1):
            dx = abs(left[0] - right[0])
            dy = abs(left[1] - right[1])
            if dx <= max_dx and min_dy <= dy <= max_dy:
                participating.update((i, j))
    if len(participating) < 2:
        return [(mask, 0)]

    peak_rows = sorted({peaks[i][1] for i in participating})
    cuts = [int(round((top + bottom) / 2))
            for top, bottom in zip(peak_rows, peak_rows[1:])
            if bottom - top >= min_dy]
    cuts = sorted({cut for cut in cuts if 0 < cut < mask.shape[0]})
    if not cuts:
        return [(mask, 0)]

    min_area = (staff_space_px ** 2) * cfg.min_area_staff_space_sq
    pieces = []
    bounds = [0] + cuts + [mask.shape[0]]
    for top, bottom in zip(bounds, bounds[1:]):
        piece = mask[top:bottom]
        if float(piece.sum()) < min_area:
            return [(mask, 0)]
        pieces.append((piece, top))
    return pieces


def mask_to_proposal(mask: np.ndarray, x_offset: float, y_offset: float, score: float,
                     method: str, staff_space_px: float, cfg: DiscoveryConfig,
                     crop_shape: Tuple[int, int]) -> Optional[Proposal]:
    """Tight box of `mask`, padded by `box_pad_staff_space` and clipped to the crop.

    The returned `Proposal.mask` covers exactly the padded box, so box and mask stay
    consistent once the mask is stored as a `MaskRle` next to the box.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    tight = mask[top:bottom, left:right]

    pad = int(round(staff_space_px * cfg.box_pad_staff_space))
    ox, oy = int(round(x_offset)), int(round(y_offset))
    x0 = max(0, ox + left - pad)
    y0 = max(0, oy + top - pad)
    x1 = min(int(crop_shape[1]), ox + right + pad)
    y1 = min(int(crop_shape[0]), oy + bottom + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    out_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    src_y0 = max(0, y0 - (oy + top))
    src_x0 = max(0, x0 - (ox + left))
    dst_y0 = max(0, (oy + top) - y0)
    dst_x0 = max(0, (ox + left) - x0)
    height = min(tight.shape[0] - src_y0, out_mask.shape[0] - dst_y0)
    width = min(tight.shape[1] - src_x0, out_mask.shape[1] - dst_x0)
    if height > 0 and width > 0:
        out_mask[dst_y0:dst_y0 + height, dst_x0:dst_x0 + width] = \
            tight[src_y0:src_y0 + height, src_x0:src_x0 + width]
    return Proposal(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0),
                    score=score, method=method, mask=out_mask)
