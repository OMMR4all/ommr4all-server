"""Adapt PCGTS point annotations and graphical connections to evaluation records."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from database.file_formats.pcgts import GraphicalConnectionType, SymbolType
from omr.discovery.config import RunConfig
from omr.discovery.schema import (Box, ReviewState, SymbolCandidate, SymbolFamily, SymbolLabel)


@dataclass(frozen=True)
class GtSymbol:
    id: str
    page: str
    block_id: str
    line_id: str
    x: float
    y: float
    family: str
    subtype: str
    connection: str


@dataclass(frozen=True)
class GtNeume:
    page: str
    line_id: str
    component_ids: List[str]


def _subtype(symbol):
    if symbol.symbol_type == SymbolType.NOTE: return symbol.note_type.name.lower()
    if symbol.symbol_type == SymbolType.CLEF: return symbol.clef_type.value
    if symbol.symbol_type == SymbolType.ACCID: return symbol.accid_type.value
    if symbol.symbol_type == SymbolType.OTHER: return symbol.symbol_class or 'unknown'
    return 'unknown'


def _connection(symbol):
    if symbol.symbol_type != SymbolType.NOTE: return 'unknown'
    return {GraphicalConnectionType.NEUME_START: 'neume_start',
            GraphicalConnectionType.LOOPED: 'looped',
            GraphicalConnectionType.GAPED: 'gaped'}[symbol.graphical_connection]


def groundtruth_symbols(page) -> List[GtSymbol]:
    out = []
    for block in page.pcgts().page.music_blocks():
        for line in block.lines:
            for symbol in line.symbols:
                if symbol.missing: continue
                out.append(GtSymbol(symbol.id, page.page, block.id, line.id,
                                    float(symbol.coord.x), float(symbol.coord.y),
                                    symbol.symbol_type.value, _subtype(symbol), _connection(symbol)))
    return out


def groundtruth_neumes(page) -> List[GtNeume]:
    out = []
    for line in page.pcgts().page.all_music_lines():
        current = []
        for symbol in line.symbols:
            if symbol.missing:
                if current and (symbol.symbol_type != SymbolType.NOTE or
                                symbol.graphical_connection == GraphicalConnectionType.NEUME_START):
                    out.append(GtNeume(page.page, line.id, current)); current = []
                continue
            if symbol.symbol_type != SymbolType.NOTE:
                if current: out.append(GtNeume(page.page, line.id, current)); current = []
                continue
            if current and symbol.graphical_connection == GraphicalConnectionType.NEUME_START:
                out.append(GtNeume(page.page, line.id, current)); current = []
            current.append(symbol.id)
        if current: out.append(GtNeume(page.page, line.id, current))
    return out


def groundtruth_candidates(page, cfg: RunConfig, run_id: str = 'groundtruth') -> List[SymbolCandidate]:
    by_line = {line.id: line for line in page.pcgts().page.all_music_lines()}
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out = []
    for gt in groundtruth_symbols(page):
        line = by_line[gt.line_id]
        ss = line.avg_line_distance(default=page.pcgts().page.avg_staff_line_distance())
        side = cfg.evaluation.derived_gt_box_staff_space * ss
        family = SymbolFamily(gt.family)
        label = SymbolLabel(family=family, subtype=gt.subtype,
                            attributes={'connection': gt.connection} if family == SymbolFamily.NOTE else {})
        out.append(SymbolCandidate(id=gt.id, run_id=run_id, book=page.book.book, page=page.page,
                                   block_id=gt.block_id, line_id=gt.line_id,
                                   box=Box(gt.x-side/2, gt.y-side/2, side, side),
                                   center_x=gt.x, center_y=gt.y, review_state=ReviewState.ACCEPTED,
                                   label=label, discovery_method='groundtruth-derived-box',
                                   discovery_score=1.0, created_at=now, updated_at=now))
    return out
