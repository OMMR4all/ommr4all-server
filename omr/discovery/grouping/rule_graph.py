"""Inspectable neume grouping baseline from geometry plus ink evidence."""
from typing import Dict, List, Tuple

import numpy as np

from omr.discovery.config import DiscoveryConfig, GroupingConfig
from omr.discovery.discovery.base import remove_staff_lines
from omr.discovery.grouping.base import NeumeGroupingMethod
from omr.discovery.regions import page_point_to_crop
from omr.discovery.schema import SymbolCandidate, SymbolFamily, SymbolRelation


class RuleGraphNeumeGrouping(NeumeGroupingMethod):
    name = 'rule_graph'

    def _eligible(self, symbols, cfg):
        return sorted([s for s in symbols if s.label.family == SymbolFamily.NOTE or
                       (cfg.accept_unknown_family and s.label.family == SymbolFamily.UNKNOWN)],
                      key=lambda s: s.center_x)

    def _evidence(self, crop, left, right, ink, labels) -> Dict[str, float]:
        ss = crop.staff_space_px
        cx1, cy1 = page_point_to_crop(crop, left.center_x, left.center_y)
        cx2, cy2 = page_point_to_crop(crop, right.center_x, right.center_y)
        dx = (cx2 - cx1) / ss
        dy = (cy2 - cy1) / ss

        # Candidate boxes and PCGTS pseudo-boxes are deliberately generous (the latter are
        # 1.4 staff spaces square), so using their edges makes every close pair overlap and
        # reduces `gap_ink` to the constant 1.0. Measure the corridor between compact,
        # staff-relative cores around the annotated centres instead. This keeps the image
        # evidence live for both discovered and GT-derived symbols.
        half = 0.18 * ss
        gap_left = int(np.ceil(cx1 + half))
        gap_right = int(np.floor(cx2 - half))
        top = max(0, int(np.floor(min(cy1, cy2) - 0.5 * ss)))
        bottom = min(ink.shape[0], int(np.ceil(max(cy1, cy2) + 0.5 * ss)))
        if gap_right <= gap_left:
            gap_ink = 1.0
        else:
            columns = ink[top:bottom, max(0, gap_left):min(ink.shape[1], gap_right)]
            gap_ink = float(columns.any(axis=0).mean()) if columns.shape[1] else 1.0

        def component_ids(cx, cy):
            radius = 0.3 * ss
            xa, ya = max(0, int(cx-radius)), max(0, int(cy-radius))
            xb, yb = min(labels.shape[1], int(np.ceil(cx+radius))), \
                     min(labels.shape[0], int(np.ceil(cy+radius)))
            return set(int(v) for v in np.unique(labels[ya:yb, xa:xb]) if v)
        shared = bool(component_ids(cx1, cy1) & component_ids(cx2, cy2))
        return {'dx': float(dx), 'dy': float(dy), 'gap_ink': gap_ink,
                'shared_cc': float(shared)}

    def _all_evidence(self, crop, symbols, cfg):
        import cv2
        ink = remove_staff_lines(crop.binary, crop.staff_lines_px, crop.staff_space_px,
                                 DiscoveryConfig())
        _, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
        return {(a.id, b.id): self._evidence(crop, a, b, ink, labels)
                for a, b in zip(symbols, symbols[1:])}

    def group(self, crop, symbols: List[SymbolCandidate], cfg: GroupingConfig) -> List[List[str]]:
        symbols = self._eligible(symbols, cfg)
        if not symbols: return []
        evidence = self._all_evidence(crop, symbols, cfg)
        groups = [[symbols[0].id]]
        for left, right in zip(symbols, symbols[1:]):
            e = evidence[(left.id, right.id)]
            linked = e['dx'] <= cfg.max_dx_staff_space and (
                cfg.geometry_only or e['gap_ink'] >= cfg.min_gap_ink or bool(e['shared_cc']))
            if linked: groups[-1].append(right.id)
            else: groups.append([right.id])
        return groups

    def relations(self, crop, symbols: List[SymbolCandidate], group: List[str],
                  cfg: GroupingConfig) -> List[SymbolRelation]:
        by_id = {s.id: s for s in symbols}
        ordered = [by_id[cid] for cid in group]
        evidence = self._all_evidence(crop, ordered, cfg)
        out = []
        for left, right in zip(ordered, ordered[1:]):
            e = evidence[(left.id, right.id)]
            out.append(SymbolRelation(left.id, right.id,
                                      'looped' if bool(e['shared_cc']) else 'gaped', e))
        return out
