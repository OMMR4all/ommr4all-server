"""Discovery by iterated normalised cuts on self-supervised patch features (MaskCut style).

One bipartition of the patch affinity graph separates "one salient thing" from the rest; doing
it repeatedly, each time masking out what was already taken, yields several objects per image.
That is the TokenCut / MaskCut recipe, with two changes that the domain forces:

* **Which side is foreground.** TokenCut assumes the corner patches of a photograph belong to
  the background. On a staff strip that prior is meaningless -- the corners are as likely to
  be ink as anything else. The side with the higher ink density (from the staff-line-removed
  mask) is taken as foreground instead.
* **Windowing.** A staff crop carries several thousand patch tokens and the dense eigen
  decomposition of the normalised Laplacian is cubic, so the crop is processed in overlapping
  column windows a few staff spaces wide. The overlap is resolved by non-maximum suppression.

This method is class agnostic: it returns regions and a saliency score, nothing else.
"""
import logging
from typing import List

import numpy as np

from omr.discovery.discovery.base import (DiscoveryContext, Proposal, SymbolDiscoveryMethod,
                                          mask_to_proposal, nms, scale_gate)
from omr.discovery.features.base import SpatialFeatureMap

logger = logging.getLogger(__name__)


class TokenCutDiscovery(SymbolDiscoveryMethod):
    name = 'tokencut'

    def discover(self, ctx: DiscoveryContext) -> List[Proposal]:
        feature_map = ctx.feature_map
        if feature_map is None:
            logger.warning('tokencut needs a feature map, skipping the line')
            return []
        crop = ctx.crop
        cfg = ctx.cfg
        gh, gw = feature_map.grid_size
        if gh < 2 or gw < 2:
            return []

        px_per_patch = feature_map.crop_pixels_per_patch()[0]
        staff_space_px = crop.staff_space_px
        window_cols = max(2, int(round(staff_space_px * cfg.tokencut_window_staff_space / px_per_patch)))
        if window_cols * gh > cfg.tokencut_max_nodes:
            window_cols = max(2, cfg.tokencut_max_nodes // gh)
        overlap_cols = min(window_cols - 1,
                           max(0, int(round(staff_space_px * cfg.tokencut_window_overlap_staff_space
                                            / px_per_patch))))
        step = max(1, window_cols - overlap_cols)

        ink_density = feature_map.area_density(ctx.ink)
        proposals: List[Proposal] = []
        for col_start in range(0, gw, step):
            col_end = min(gw, col_start + window_cols)
            if col_end - col_start < 2:
                break
            proposals.extend(self._cut_window(ctx, feature_map, ink_density, col_start, col_end))
            if col_end >= gw:
                break

        proposals = scale_gate(proposals, staff_space_px, cfg)
        return nms(proposals, cfg.nms_iou)

    def _cut_window(self, ctx: DiscoveryContext, feature_map: SpatialFeatureMap,
                    ink_density: np.ndarray, col_start: int, col_end: int) -> List[Proposal]:
        import cv2
        from scipy.linalg import eigh

        cfg = ctx.cfg
        gh = feature_map.grid_size[0]
        window = feature_map.features[:, col_start:col_end]
        n_cols = window.shape[1]
        flat = window.reshape(-1, window.shape[-1]).astype(np.float64)
        n_nodes = flat.shape[0]
        ink_flat = ink_density[:, col_start:col_end].reshape(-1)

        similarity = flat @ flat.T
        affinity = np.where(similarity > cfg.tokencut_tau, 1.0, cfg.tokencut_eps)

        valid = feature_map.area_density(
            np.ones(feature_map.crop_size, dtype=bool))[:, col_start:col_end].reshape(-1) >= 0.5
        remaining = valid.copy()
        proposals: List[Proposal] = []
        min_ink = (ctx.crop.staff_space_px ** 2) * cfg.min_ink_area_staff_space_sq
        for _ in range(max(1, cfg.tokencut_n_cuts)):
            index = np.flatnonzero(remaining)
            if index.size < 4:
                break
            sub = affinity[np.ix_(index, index)]
            degree = sub.sum(axis=1)
            inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
            laplacian = np.eye(index.size) - (inv_sqrt[:, None] * sub * inv_sqrt[None, :])
            try:
                _, vectors = eigh(laplacian, subset_by_index=[1, 1])
            except Exception as e:  # pragma: no cover - LAPACK failure on a degenerate window
                logger.info('Ncut failed on a window: %s', e)
                break
            fiedler = vectors[:, 0]
            threshold = float(fiedler.mean())
            side = fiedler > threshold
            if side.all() or not side.any():
                break
            ink_side = ink_flat[index]
            foreground = side if ink_side[side].mean() > ink_side[~side].mean() else ~side
            if not foreground.any():
                break

            deviation = np.abs(fiedler - threshold)
            scale = float(deviation.max()) or 1.0
            deviation = deviation / scale

            patch_mask = np.zeros(n_nodes, dtype=bool)
            patch_mask[index[foreground]] = True
            node_score = np.zeros(n_nodes, dtype=float)
            node_score[index] = deviation
            grid_mask = patch_mask.reshape(gh, n_cols)

            # MaskCut keeps one object per cut: the connected component of the foreground that
            # holds the most salient patch. Keeping the whole half of the bipartition instead
            # shatters into dozens of patch specks, because the split is by construction
            # roughly balanced while a symbol covers a few patches.
            n_labels, labels = cv2.connectedComponents(grid_mask.astype(np.uint8), connectivity=8)
            if n_labels < 2:
                break
            seed = int(np.argmax(np.where(patch_mask, node_score, -np.inf)))
            label = int(labels.reshape(-1)[seed])
            if label == 0:
                break
            component = labels == label
            remaining[index] &= ~component.reshape(-1)[index]

            score = float(node_score.reshape(gh, n_cols)[component].mean())
            full = np.zeros(feature_map.grid_size, dtype=bool)
            full[:, col_start:col_end] = component
            pixel_mask = feature_map.grid_to_crop(full, nearest=True) > 0
            if not pixel_mask.any():
                continue
            if float((pixel_mask & ctx.ink).sum()) < min_ink:
                continue
            proposal = mask_to_proposal(pixel_mask, 0, 0, score, self.name,
                                        ctx.crop.staff_space_px, cfg, ctx.ink.shape[:2])
            if proposal is not None:
                proposals.append(proposal)
        return proposals
