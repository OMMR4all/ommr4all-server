"""Fused appearance + structure descriptors of neume groups."""
from typing import Dict, Tuple

import numpy as np

from omr.discovery.config import NeumeEmbeddingConfig
from omr.discovery.features.base import l2_normalize
from omr.discovery.regions import page_box_to_crop
from omr.discovery.schema import NeumeCandidate


def structural_vector(store, n: NeumeCandidate, staff_space_page: float) -> np.ndarray:
    components = [store.candidates[cid] for cid in n.component_ids]
    dx = np.array([(b.center_x - a.center_x) / max(staff_space_page, 1e-12)
                   for a, b in zip(components, components[1:])], dtype=float)
    dy = np.array([(b.center_y - a.center_y) / max(staff_space_page, 1e-12)
                   for a, b in zip(components, components[1:])], dtype=float)
    signs = [float(np.sign(v)) for v in dy[:3]] + [0.0] * max(0, 3 - len(dy))
    looped = sum(r.kind == 'looped' for r in n.relations) / max(1, len(n.relations))
    vector = np.array([
        len(components) / 4.0,
        float(dx.mean()) if len(dx) else 0.0,
        float(dx.std()) if len(dx) else 0.0,
        float(dy.mean()) if len(dy) else 0.0,
        float(dy.std()) if len(dy) else 0.0,
        *signs[:3],
        float(np.log(max(n.box.w, 1e-12) / max(n.box.h, 1e-12))),
        float(np.log(max(n.box.w, 1e-12) / max(staff_space_page, 1e-12))),
        float(looped),
    ], dtype=np.float32)
    return l2_normalize(vector)


def embed_neumes(store, crops: Dict[Tuple[str, str], object], extractor,
                 cfg: NeumeEmbeddingConfig) -> np.ndarray:
    feature_maps = {}
    rows = []
    for index, n in enumerate(store.ordered_neumes()):
        key = (n.page, n.line_id)
        crop = crops[key]
        if key not in feature_maps:
            feature_maps[key] = extractor.extract_feature_map(crop.image, crop.staff_space_px)
        x, y, w, h = page_box_to_crop(crop, n.box)
        appearance = feature_maps[key].pool_box(x, y, w, h)
        structure = structural_vector(store, n, crop.staff_space_page)
        fused = np.concatenate([cfg.appearance_weight * appearance,
                                cfg.structure_weight * structure]).astype(np.float32)
        rows.append(l2_normalize(fused))
        n.embedding_index = index
    return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
