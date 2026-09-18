"""Review operations for symbols and neume groups.

Manual field paths survive bulk operations. Review state and semantic interpretation stay
orthogonal: accepted + unknown and unreviewed + fully labelled are both valid.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from omr.discovery.schema import (Box, GroupingState, MaskRle, NeumeCandidate, RejectionReason,
                                  ReviewState, SymbolCandidate, SymbolLabel, SymbolRelation,
                                  validate_label, validate_neume_type)
from omr.discovery.store import CandidateStore, Interaction


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mark_manual(candidate, path: str) -> None:
    if path not in candidate.manual_fields:
        candidate.manual_fields.append(path)


def accept(c: SymbolCandidate, label: Optional[SymbolLabel] = None) -> None:
    c.review_state = ReviewState.ACCEPTED
    c.rejection_reason = None
    if label is not None:
        errors = validate_label(label)
        if errors:
            raise ValueError('; '.join(errors))
        c.label = label
    c.updated_at = _now()


def reject(c: SymbolCandidate, reason: RejectionReason) -> None:
    c.review_state = ReviewState.REJECTED
    c.rejection_reason = reason
    c.updated_at = _now()


def modify_geometry(c: SymbolCandidate, box: Box, mask: Optional[MaskRle] = None) -> None:
    c.box = box
    c.center_x, c.center_y = box.center()
    c.mask_rle = mask
    c.review_state = ReviewState.MODIFIED
    _mark_manual(c, 'box')
    c.updated_at = _now()


def set_label(c: SymbolCandidate, *, family=None, subtype=None, attributes=None,
              manual: bool = False) -> None:
    new = SymbolLabel(family=family if family is not None else c.label.family,
                      subtype=subtype if subtype is not None else c.label.subtype,
                      attributes=dict(c.label.attributes))
    if attributes is not None:
        new.attributes.update(attributes)
    errors = validate_label(new)
    if errors:
        raise ValueError('; '.join(errors))
    c.label = new
    if manual:
        if family is not None: _mark_manual(c, 'label.family')
        if subtype is not None: _mark_manual(c, 'label.subtype')
        for key in (attributes or {}): _mark_manual(c, 'label.attributes.' + key)
    c.updated_at = _now()


def bulk_apply(store: CandidateStore, candidate_ids: List[str], *, review_state=None,
               family=None, subtype=None, attributes=None, reason=None, scope: str,
               respect_overrides: bool = True) -> int:
    affected = 0
    for cid in candidate_ids:
        c = store.candidates[cid]
        changed = False
        if review_state is not None and (not respect_overrides or 'review_state' not in c.manual_fields):
            c.review_state = review_state
            c.rejection_reason = reason if review_state == ReviewState.REJECTED else None
            changed = True
        kwargs = {}
        if family is not None and (not respect_overrides or 'label.family' not in c.manual_fields):
            kwargs['family'] = family
        if subtype is not None and (not respect_overrides or 'label.subtype' not in c.manual_fields):
            kwargs['subtype'] = subtype
        kept_attrs = {k: v for k, v in (attributes or {}).items()
                      if not respect_overrides or 'label.attributes.' + k not in c.manual_fields}
        if kept_attrs: kwargs['attributes'] = kept_attrs
        if kwargs:
            set_label(c, **kwargs)
            changed = True
        if changed:
            c.updated_at = _now()
            affected += 1
    store.log_interaction(Interaction(ts=_now(), op='bulk_apply', scope=scope,
                                      n_affected=affected,
                                      detail={'respect_overrides': str(respect_overrides)}))
    return affected


def confirm_group(n: NeumeCandidate) -> None:
    n.grouping_state = GroupingState.CONFIRMED
    n.updated_at = _now()


def reject_group(n: NeumeCandidate, reason: RejectionReason) -> None:
    n.grouping_state = GroupingState.REJECTED
    n.attributes['rejection_reason'] = reason.value
    n.updated_at = _now()


def set_neume_type(n: NeumeCandidate, neume_type: str, *, manual: bool = False) -> None:
    errors = validate_neume_type(neume_type)
    if errors: raise ValueError('; '.join(errors))
    n.neume_type = neume_type
    if manual: _mark_manual(n, 'neume_type')
    n.updated_at = _now()


def _rebuild(store: CandidateStore, n: NeumeCandidate, old_relations=None) -> None:
    old_relations = old_relations if old_relations is not None else n.relations
    old = {(r.from_id, r.to_id): r for r in old_relations}
    n.component_ids = sorted(dict.fromkeys(n.component_ids), key=lambda cid: store.candidates[cid].center_x)
    if not n.component_ids:
        raise ValueError('a neume group cannot have zero components')
    box = store.candidates[n.component_ids[0]].box
    for cid in n.component_ids[1:]: box = box.union(store.candidates[cid].box)
    n.box = box
    n.relations = [old.get((a, b), SymbolRelation(a, b, 'gaped', {}))
                   for a, b in zip(n.component_ids, n.component_ids[1:])]
    for cid in n.component_ids: store.candidates[cid].neume_id = n.id
    n.grouping_state = GroupingState.MODIFIED
    n.updated_at = _now()


def add_component(store: CandidateStore, n: NeumeCandidate, candidate_id: str) -> None:
    c = store.candidates[candidate_id]
    if (c.book,c.page,c.line_id) != (n.book,n.page,n.line_id):
        raise ValueError('component and group must be on the same page and line')
    if c.neume_id is not None and c.neume_id != n.id:
        raise ValueError('candidate already belongs to another group')
    n.component_ids.append(candidate_id)
    _rebuild(store, n)


def remove_component(store: CandidateStore, n: NeumeCandidate, candidate_id: str) -> None:
    if candidate_id not in n.component_ids: raise ValueError('candidate is not in the group')
    if len(n.component_ids) == 1: raise ValueError('a neume group cannot have zero components')
    old = list(n.relations)
    n.component_ids.remove(candidate_id)
    store.candidates[candidate_id].neume_id = None
    _rebuild(store, n, old)


def _copy_group(n, component_ids, new_id):
    now = _now()
    return NeumeCandidate(id=new_id, grouping_run_id=n.grouping_run_id,
                          symbols_run_id=n.symbols_run_id, book=n.book, page=n.page,
                          line_id=n.line_id, box=n.box, component_ids=list(component_ids),
                          neume_type=n.neume_type if n.neume_type == 'unknown' else 'unknown',
                          created_at=now, updated_at=now)


def split_group(store: CandidateStore, n: NeumeCandidate, at_index: int) -> Tuple[NeumeCandidate, NeumeCandidate]:
    if not 0 < at_index < len(n.component_ids): raise ValueError('split index must divide the group')
    left = _copy_group(n, n.component_ids[:at_index], uuid4().hex)
    right = _copy_group(n, n.component_ids[at_index:], uuid4().hex)
    del store.neumes[n.id]
    for cid in n.component_ids: store.candidates[cid].neume_id = None
    store.neumes[left.id] = left; store.neumes[right.id] = right
    _rebuild(store, left); _rebuild(store, right)
    return left, right


def merge_groups(store: CandidateStore, neume_ids: List[str]) -> NeumeCandidate:
    if not neume_ids: raise ValueError('no groups to merge')
    groups = [store.neumes[nid] for nid in neume_ids]
    if len({(n.page, n.line_id) for n in groups}) != 1: raise ValueError('groups must be on the same line')
    first = groups[0]
    merged = _copy_group(first, sum((n.component_ids for n in groups), []), uuid4().hex)
    merged.neume_type = 'unknown'
    old_relations = sum((n.relations for n in groups), [])
    for n in groups: del store.neumes[n.id]
    store.neumes[merged.id] = merged
    _rebuild(store, merged, old_relations)
    return merged
