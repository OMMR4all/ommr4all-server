"""Registry of symbol discovery methods."""
from typing import Dict, List, Type

from omr.discovery.config import DiscoveryConfig
from omr.discovery.discovery.base import (DiscoveryContext, Proposal, SymbolDiscoveryMethod,
                                          nms, remove_staff_lines, scale_gate, split_wide)
from omr.discovery.discovery.ink_components import InkComponentDiscovery
from omr.discovery.discovery.patch_peaks import PatchPeakDiscovery
from omr.discovery.discovery.tokencut import TokenCutDiscovery

SYMBOL_DISCOVERY_METHODS: Dict[str, Type[SymbolDiscoveryMethod]] = {
    InkComponentDiscovery.name: InkComponentDiscovery,
    PatchPeakDiscovery.name: PatchPeakDiscovery,
    TokenCutDiscovery.name: TokenCutDiscovery,
}


def build_methods(cfg: DiscoveryConfig) -> List[SymbolDiscoveryMethod]:
    """`'ink+tokencut'` -> both methods; the caller unions their proposals by NMS."""
    methods = []
    for name in (n.strip() for n in cfg.method.split('+')):
        if not name:
            continue
        try:
            methods.append(SYMBOL_DISCOVERY_METHODS[name]())
        except KeyError:
            raise ValueError('unknown discovery method {!r} (known: {})'.format(
                name, ', '.join(sorted(SYMBOL_DISCOVERY_METHODS))))
    if not methods:
        raise ValueError('discovery.method {!r} selects no method'.format(cfg.method))
    return methods


def needs_features(cfg: DiscoveryConfig) -> bool:
    """True if any selected method reads the patch feature map."""
    return any(isinstance(m, (TokenCutDiscovery, PatchPeakDiscovery)) for m in build_methods(cfg))


__all__ = ['SYMBOL_DISCOVERY_METHODS', 'build_methods', 'needs_features', 'DiscoveryContext',
           'Proposal', 'SymbolDiscoveryMethod', 'InkComponentDiscovery', 'PatchPeakDiscovery',
           'TokenCutDiscovery',
           'nms', 'remove_staff_lines', 'scale_gate', 'split_wide']
