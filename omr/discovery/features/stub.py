"""A weights-free, deterministic feature extractor.

Exists so that the whole pipeline -- discovery, clustering, grouping, evaluation, reporting --
can be exercised in the test suite and in a smoke run without downloading any model. The
descriptors are hand-made statistics of the patch pixels, not learned features, so they are
useless for real clustering; they are a stand-in for the interface, never a fallback for a
real run.
"""
from typing import Any, Dict

import numpy as np

from omr.discovery.config import FeatureConfig
from omr.discovery.features.base import SpatialFeatureMap, SymbolFeatureExtractor, l2_normalize

PATCH_SIZE = 8
DIM = 16


class StubFeatureExtractor(SymbolFeatureExtractor):
    name = 'stub'

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.patch_size = PATCH_SIZE

    def extract_feature_map(self, image: np.ndarray, staff_space_px: float) -> SpatialFeatureMap:
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            gray = image.astype(float)
        gray = gray / 255.0
        h, w = gray.shape
        ps = self.patch_size
        gh = max(1, (h + ps - 1) // ps)
        gw = max(1, (w + ps - 1) // ps)
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        padded = np.full((gh * ps, gw * ps), np.median(border), dtype=gray.dtype)
        padded[:h, :w] = gray
        blocks = padded.reshape(gh, ps, gw, ps).transpose(0, 2, 1, 3)

        ink = (blocks < 0.5).astype(float)
        half = ps // 2
        third = max(1, ps // 3)
        quad_a = blocks[:, :, :half, :half].mean(axis=(2, 3))
        quad_b = blocks[:, :, :half, half:].mean(axis=(2, 3))
        quad_c = blocks[:, :, half:, :half].mean(axis=(2, 3))
        quad_d = blocks[:, :, half:, half:].mean(axis=(2, 3))
        band_top = blocks[:, :, :third, :].mean(axis=(2, 3))
        band_mid = blocks[:, :, third:2 * third, :].mean(axis=(2, 3))
        band_bot = blocks[:, :, 2 * third:, :].mean(axis=(2, 3))
        col_left = blocks[:, :, :, :third].mean(axis=(2, 3))
        col_mid = blocks[:, :, :, third:2 * third].mean(axis=(2, 3))
        col_right = blocks[:, :, :, 2 * third:].mean(axis=(2, 3))
        centre = blocks[:, :, third:ps - third, third:ps - third].mean(axis=(2, 3)) \
            if ps - 2 * third > 0 else blocks.mean(axis=(2, 3))

        rows = (np.arange(gh, dtype=float) / max(1, gh - 1)).reshape(gh, 1) * np.ones((1, gw))
        cols = (np.arange(gw, dtype=float) / max(1, gw - 1)).reshape(1, gw) * np.ones((gh, 1))

        channels = [
            blocks.mean(axis=(2, 3)),
            blocks.std(axis=(2, 3)),
            ink.mean(axis=(2, 3)),
            rows,
            cols,
            (quad_a + quad_b) - (quad_c + quad_d),
            (quad_a + quad_c) - (quad_b + quad_d),
            (quad_a + quad_d) - (quad_b + quad_c),
            band_top - band_mid,
            band_mid - band_bot,
            col_left - col_mid,
            col_mid - col_right,
            centre - blocks.mean(axis=(2, 3)),
            np.abs(np.diff(blocks, axis=3)).mean(axis=(2, 3)),
            np.abs(np.diff(blocks, axis=2)).mean(axis=(2, 3)),
            blocks.max(axis=(2, 3)) - blocks.min(axis=(2, 3)),
        ]
        assert len(channels) == DIM
        features = l2_normalize(np.stack(channels, axis=-1).astype(np.float32))
        return SpatialFeatureMap(features=features, patch_size=ps,
                                 input_size=(gh * ps, gw * ps), content_size=(h, w),
                                 crop_size=(h, w))

    def describe(self) -> Dict[str, Any]:
        return {
            'backend': self.name,
            'model_name': 'stub',
            'model_revision': None,
            'patch_size': self.patch_size,
            'embedding_dim': DIM,
            'num_prefix_tokens': 0,
            'device': 'cpu',
            'note': 'hand made pixel statistics, no learned weights; for tests and smoke runs',
        }
