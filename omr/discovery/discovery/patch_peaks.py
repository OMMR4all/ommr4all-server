"""CC-free symbol centre proposals from patch appearance novelty and staff-removed ink."""
import logging
from typing import List, Tuple

import numpy as np

from omr.discovery.discovery.base import (DiscoveryContext, Proposal, SymbolDiscoveryMethod,
                                          scale_gate)

logger = logging.getLogger(__name__)


class PatchPeakDiscovery(SymbolDiscoveryMethod):
    name = 'patch_peaks'

    def discover(self, ctx: DiscoveryContext) -> List[Proposal]:
        feature_map = ctx.feature_map
        if feature_map is None:
            logger.warning('patch_peaks needs a feature map, skipping the line')
            return []
        if not ctx.ink.any():
            return []

        gh, gw = feature_map.grid_size
        if gh == 0 or gw == 0:
            return []

        ink_density = feature_map.area_density(ctx.ink)
        region_density = feature_map.area_density(ctx.crop.region_mask)
        valid = region_density >= 0.5
        if not valid.any():
            return []

        cfg = ctx.cfg
        k = max(1, int(cfg.patch_peak_min_background_patches))
        background = valid & (ink_density <= cfg.patch_peak_background_max_ink_density)
        if int(background.sum()) >= k:
            prototype_sum = np.einsum('rcd,rc->d', feature_map.features, background,
                                      optimize=True)
            prototype_count = int(background.sum())
        else:
            valid_cells = sorted(zip(*np.nonzero(valid)),
                                 key=lambda cell: (float(ink_density[cell]),
                                                   cell[0], cell[1]))
            fallback_cells = valid_cells[:min(k, len(valid_cells))]
            prototype_sum = np.zeros(feature_map.dim, dtype=np.float64)
            for cell in fallback_cells:
                prototype_sum += feature_map.features[cell]
            prototype_count = len(fallback_cells)
        if prototype_count == 0:
            return []
        prototype = prototype_sum / prototype_count
        prototype_norm = float(np.linalg.norm(prototype))
        if prototype_norm < 1e-12:
            return []
        global_prototype = prototype / prototype_norm

        supported = valid & (ink_density >= cfg.patch_peak_min_ink_density)
        raw_scores = np.zeros((gh, gw), dtype=np.float32)
        local_half_size = (cfg.patch_peak_local_background_staff_space
                           * ctx.crop.staff_space_px)
        patch_width, patch_height = feature_map.crop_pixels_per_patch()
        local_col_radius = max(0, int(np.ceil(local_half_size / patch_width)))
        local_row_radius = max(0, int(np.ceil(local_half_size / patch_height)))
        for row, col in zip(*np.nonzero(supported)):
            row_lo = max(0, row - local_row_radius)
            row_hi = min(gh, row + local_row_radius + 1)
            col_lo = max(0, col - local_col_radius)
            col_hi = min(gw, col + local_col_radius + 1)
            local_background = background[row_lo:row_hi, col_lo:col_hi]
            local_count = int(local_background.sum())
            local_prototype = global_prototype
            if local_count >= k:
                local_sum = np.einsum(
                    'rcd,rc->d',
                    feature_map.features[row_lo:row_hi, col_lo:col_hi],
                    local_background,
                    optimize=True)
                local_mean = local_sum / local_count
                local_norm = float(np.linalg.norm(local_mean))
                if local_norm >= 1e-12:
                    local_prototype = local_mean / local_norm
            similarity = float(np.dot(feature_map.features[row, col], local_prototype))
            novelty = float(np.clip((1.0 - similarity) / 2.0, 0.0, 1.0))
            raw_scores[row, col] = novelty * float(ink_density[row, col])

        absolutely_eligible = supported & (raw_scores >= cfg.patch_peak_min_score)
        supported_cells = list(zip(*np.nonzero(supported)))
        centers = {cell: feature_map.patch_center(*cell) for cell in supported_cells}
        prominence_radius = (cfg.patch_peak_prominence_staff_space
                             * ctx.crop.staff_space_px)
        min_distance = cfg.patch_peak_min_distance_staff_space * ctx.crop.staff_space_px
        survivors: List[Tuple[float, float, int, int, float]] = []
        for row, col in zip(*np.nonzero(absolutely_eligible)):
            center_x, center_y = centers[(row, col)]
            neighbor_scores = []
            has_strictly_higher = False
            for neighbor in supported_cells:
                if neighbor == (row, col):
                    continue
                other_x, other_y = centers[neighbor]
                distance = float(np.hypot(center_x - other_x, center_y - other_y))
                neighbor_score = float(raw_scores[neighbor])
                if distance <= prominence_radius:
                    neighbor_scores.append(neighbor_score)
                if distance <= min_distance and neighbor_score > float(raw_scores[row, col]):
                    has_strictly_higher = True
            baseline = float(np.median(neighbor_scores)) if neighbor_scores else 0.0
            prominence = float(raw_scores[row, col]) - baseline
            if prominence < cfg.patch_peak_min_prominence or has_strictly_higher:
                continue
            survivors.append((prominence, float(raw_scores[row, col]), row, col, baseline))
        survivors.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

        crop_height, crop_width = ctx.ink.shape[:2]
        target_half_size = cfg.patch_peak_box_staff_space * ctx.crop.staff_space_px / 2.0
        pixel_response = feature_map.grid_to_crop(raw_scores)
        retained = []
        for prominence, raw_score, row, col, baseline in survivors:
            seed_x, seed_y = feature_map.patch_center(row, col)
            center = (seed_x, seed_y)
            radius = target_half_size
            if radius > 0:
                x0 = max(0, int(np.floor(seed_x - radius)))
                y0 = max(0, int(np.floor(seed_y - radius)))
                x1 = min(crop_width, int(np.ceil(seed_x + radius)))
                y1 = min(crop_height, int(np.ceil(seed_y + radius)))
                if x1 > x0 and y1 > y0:
                    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
                    ys = np.arange(y0, y1, dtype=np.float64) + 0.5
                    delta_x = xs[None, :] - seed_x
                    delta_y = ys[:, None] - seed_y
                    distance_squared = delta_x * delta_x + delta_y * delta_y
                    circular = distance_squared <= radius * radius
                    positive_response = np.maximum(
                        pixel_response[y0:y1, x0:x1] - baseline, 0.0)
                    sigma = radius / 2.0
                    weights = (ctx.ink[y0:y1, x0:x1] * positive_response * circular
                               * np.exp(-distance_squared / (2.0 * sigma * sigma)))
                    total_weight = float(weights.sum())
                    if np.isfinite(total_weight) and total_weight > 0:
                        refined_x = float((weights * xs[None, :]).sum() / total_weight)
                        refined_y = float((weights * ys[:, None]).sum() / total_weight)
                        if np.isfinite(refined_x) and np.isfinite(refined_y):
                            center = (refined_x, refined_y)
            if all(np.hypot(center[0] - other[0], center[1] - other[1]) >= min_distance
                   for other, _, _, _, _, _ in retained):
                retained.append((center, prominence, raw_score, row, col, baseline))

        min_ink = cfg.min_ink_area_staff_space_sq * ctx.crop.staff_space_px ** 2
        proposals = []
        for (center_x, center_y), _, raw_score, _, _, _ in retained:
            half_size = min(target_half_size, center_x, center_y,
                            crop_width - center_x, crop_height - center_y)
            if half_size <= 0:
                continue
            x = center_x - half_size
            y = center_y - half_size
            side = 2.0 * half_size
            x0 = max(0, int(np.floor(x)))
            y0 = max(0, int(np.floor(y)))
            x1 = min(crop_width, int(np.ceil(x + side)))
            y1 = min(crop_height, int(np.ceil(y + side)))
            if float(ctx.ink[y0:y1, x0:x1].sum()) < min_ink:
                continue
            proposals.append(Proposal(x=x, y=y, w=side, h=side,
                                      score=raw_score, method=self.name))
        return scale_gate(proposals, ctx.crop.staff_space_px, cfg)

