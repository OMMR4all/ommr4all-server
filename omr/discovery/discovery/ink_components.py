"""Discovery by connected components of the ink after staff line removal.

The manuscript-specific baseline: a chant manuscript is bitonal, the staff lines are the only
long thin horizontal structures, and every symbol is a compact blob of ink. Removing the lines
and taking connected components therefore already localises most symbols, and the scale prior
(everything measured in staff spaces) rejects specks and marginalia.

It knows nothing about symbol classes and it does not learn anything -- which is exactly why
it is the reference that the feature based method has to beat.
"""
from typing import List

import numpy as np

from omr.discovery.discovery.base import (DiscoveryContext, Proposal, SymbolDiscoveryMethod,
                                          mask_to_proposal, scale_gate, split_stacked, split_wide)
from omr.discovery.regions import page_point_to_crop

class InkComponentDiscovery(SymbolDiscoveryMethod):
    name = 'ink'

    def discover(self, ctx: DiscoveryContext) -> List[Proposal]:
        import cv2
        crop = ctx.crop
        staff_space_px = crop.staff_space_px
        ink = ctx.ink
        if not ink.any():
            return []

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8)
        min_area = (staff_space_px ** 2) * ctx.cfg.min_area_staff_space_sq
        max_height = staff_space_px * ctx.cfg.max_height_staff_space
        max_width = staff_space_px * ctx.cfg.max_width_staff_space
        reference_area = (staff_space_px ** 2) * 0.6
        line_points = np.asarray(getattr(crop.line.coords, 'points', []), dtype=float)
        if line_points.size:
            line_left = min(page_point_to_crop(crop, float(x), float(y))[0]
                            for x, y in line_points)
        else:
            allowed_columns = np.flatnonzero(crop.region_mask.any(axis=0))
            line_left = int(allowed_columns[0]) if len(allowed_columns) else 0

        proposals: List[Proposal] = []
        for label in range(1, n_labels):
            left, top, width, height, area = stats[label]
            if area < min_area or height > max_height:
                continue
            component = labels[top:top + height, left:left + width] == label
            horizontal_pieces = split_wide(component, staff_space_px, ctx.cfg) \
                if width > max_width else [(component, 0)]
            for horizontal, x_offset in horizontal_pieces:
                absolute_left = left + x_offset
                far_from_left = absolute_left - line_left >= (
                    staff_space_px * ctx.cfg.stacked_split_ignore_left_staff_space)
                vertical_pieces = split_stacked(horizontal, staff_space_px, ctx.cfg) \
                    if far_from_left else [(horizontal, 0)]
                for piece, y_offset in vertical_pieces:
                    ink_area = float(piece.sum())
                    if ink_area < min_area:
                        continue
                    score = float(min(1.0, ink_area / reference_area))
                    proposal = mask_to_proposal(
                        piece, absolute_left, top + y_offset, score, self.name,
                        staff_space_px, ctx.cfg, ink.shape[:2])
                    if proposal is not None:
                        proposals.append(proposal)
        return scale_gate(proposals, staff_space_px, ctx.cfg)
