"""The discovery ontology and its serialisable data model.

Four concerns are kept in four separate places and are never collapsed into one flat class
enumeration:

1. *graphical symbol localisation*  -> `SymbolCandidate.box` / `.mask_rle` / `.discovery_score`
2. *semantic symbol interpretation* -> `SymbolCandidate.label` (`SymbolLabel`)
3. *grouping into neumes*           -> `NeumeCandidate.component_ids` / `.relations`
4. *semantic neume interpretation*  -> `NeumeCandidate.neume_type`

Consequences that the rest of the package relies on:

* `UNKNOWN` is "not interpreted yet", never "background". A candidate can be localised and
  accepted while its family is still unknown.
* `ReviewState.UNREVIEWED` is not a negative example, and a rejected candidate only counts as
  background evidence when its `RejectionReason.is_background_evidence()` says so.
* A cluster id is a similarity label. It carries no semantics and is stored separately from
  `label`.
* A neume type is a property of the group, not a sub type of a note; the components of a
  group keep their own labels and are never discarded.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin

from database.file_formats.pcgts.page.musicsymbol import AccidType, BasicNeumeType, ClefType, NoteType, SymbolType

#: The sentinel for "not interpreted yet". Deliberately not a member of any closed enum of
#: semantic classes, and deliberately different from "background"/"not a symbol".
UNKNOWN = 'unknown'


class SymbolFamily(Enum):
    """The coarse kind of a candidate.

    The values of the four musical families mirror `SymbolType` verbatim so that converting an
    accepted candidate into a `MusicSymbol` later is a one-to-one mapping. `UNKNOWN` and
    `NOT_A_SYMBOL` have no `SymbolType` counterpart on purpose: the first is an un-interpreted
    candidate, the second an explicit statement that the region is not a music symbol.
    """
    UNKNOWN = 'unknown'
    NOTE = SymbolType.NOTE.value
    CLEF = SymbolType.CLEF.value
    ACCID = SymbolType.ACCID.value
    OTHER = SymbolType.OTHER.value
    NOT_A_SYMBOL = 'not_a_symbol'

    def to_symbol_type(self) -> Optional[SymbolType]:
        """The PCGTS `SymbolType` of this family, or None if it has no counterpart."""
        try:
            return SymbolType(self.value)
        except ValueError:
            return None


class ReviewState(Enum):
    UNREVIEWED = 'unreviewed'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    MODIFIED = 'modified'


class RejectionReason(Enum):
    ARTIFACT = 'artifact'
    DUPLICATE = 'duplicate'
    FRAGMENT = 'fragment'
    MERGED = 'merged_objects'
    WRONG_GEOMETRY = 'wrong_geometry'
    NOT_A_MUSIC_SYMBOL = 'not_a_music_symbol'
    STAFF_LINE = 'staff_line'

    def is_background_evidence(self) -> bool:
        """True only when the rejection states that there is no symbol at that place.

        A duplicate, a fragment or a merged pair of symbols all *do* sit on a symbol: using
        them as negative examples would teach a later refinement step to ignore real ink.
        """
        return self in (RejectionReason.NOT_A_MUSIC_SYMBOL, RejectionReason.STAFF_LINE)


class GroupingState(Enum):
    UNGROUPED = 'ungrouped'
    PROPOSED = 'proposed'
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    MODIFIED = 'modified'


# ----------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------
@dataclass
class Box(DataClassJSONMixin):
    """An axis aligned box in height-normalised page coordinates.

    Same wire shape as the boxes of the melodic pattern matcher
    (`omr/steps/tools/symbol_pattern_matching/predictor.py`), i.e. `x`/`y` is the top left
    corner and all four values are divided by the *image height*.
    """
    x: float
    y: float
    w: float
    h: float

    def right(self) -> float:
        return self.x + self.w

    def bottom(self) -> float:
        return self.y + self.h

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right() and self.y <= y <= self.bottom()

    def intersection_area(self, other: 'Box') -> float:
        dx = min(self.right(), other.right()) - max(self.x, other.x)
        dy = min(self.bottom(), other.bottom()) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: 'Box') -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        return inter / (self.area() + other.area() - inter)

    def union(self, other: 'Box') -> 'Box':
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        return Box(x, y, max(self.right(), other.right()) - x, max(self.bottom(), other.bottom()) - y)


@dataclass
class MaskRle(DataClassJSONMixin):
    """A binary mask of the size of its box, run-length encoded row major.

    `runs` always starts with the length of a background run (possibly 0) and then alternates
    foreground/background.
    """
    width: int
    height: int
    runs: List[int] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        out = np.zeros(self.width * self.height, dtype=bool)
        pos = 0
        foreground = False
        for run in self.runs:
            if foreground and run > 0:
                out[pos:pos + run] = True
            pos += run
            foreground = not foreground
        return out.reshape((self.height, self.width))

    @staticmethod
    def from_array(a: np.ndarray) -> 'MaskRle':
        a = np.asarray(a).astype(bool)
        height, width = a.shape[:2]
        flat = a.reshape(-1)
        if flat.size == 0:
            return MaskRle(width=width, height=height, runs=[])
        changes = np.flatnonzero(np.diff(flat)) + 1
        bounds = np.concatenate(([0], changes, [flat.size]))
        lengths = np.diff(bounds).astype(int).tolist()
        runs = [0] if bool(flat[0]) else []
        runs.extend(lengths)
        return MaskRle(width=width, height=height, runs=runs)


# ----------------------------------------------------------------------------------------
# semantic interpretation of a single symbol
# ----------------------------------------------------------------------------------------
#: `family -> allowed sub types`. The sub type of a family is exactly the concrete PCGTS sub
#: type enum of that family, plus `UNKNOWN` for "family decided, sub type still open".
#: `SymbolFamily.OTHER` is refined by a registered `SymbolClass.id` which only exists in the
#: database, so any non-empty string is accepted there.
SUBTYPE_VOCABULARY: Dict[SymbolFamily, Optional[List[str]]] = {
    SymbolFamily.UNKNOWN: [UNKNOWN],
    SymbolFamily.NOT_A_SYMBOL: [UNKNOWN],
    SymbolFamily.NOTE: [UNKNOWN] + [t.name.lower() for t in NoteType],
    SymbolFamily.CLEF: [UNKNOWN] + [t.value for t in ClefType],
    SymbolFamily.ACCID: [UNKNOWN] + [t.value for t in AccidType],
    SymbolFamily.OTHER: None,  # free form: a registered symbol class id
}

#: Independent properties of a symbol. Keeping them in their own key/value pairs is what
#: prevents a combined enumeration such as "virga liquescent at neume start".
ATTRIBUTE_VOCABULARY: Dict[str, List[str]] = {
    # mirrors GraphicalConnectionType; 'unknown' means the relation was not decided
    'connection': [UNKNOWN, 'neume_start', 'gaped', 'looped'],
    'liquescent': [UNKNOWN, 'true', 'false'],
}


@dataclass
class SymbolLabel(DataClassJSONMixin):
    family: SymbolFamily = SymbolFamily.UNKNOWN
    subtype: str = UNKNOWN
    attributes: Dict[str, str] = field(default_factory=dict)


def validate_label(label: SymbolLabel) -> List[str]:
    """Human readable reasons why `label` is not a valid interpretation; empty == valid."""
    errors: List[str] = []
    allowed = SUBTYPE_VOCABULARY.get(label.family, None)
    if allowed is None:
        if not isinstance(label.subtype, str) or not label.subtype:
            errors.append('subtype of family {} must be a non-empty string'.format(label.family.value))
    elif label.subtype not in allowed:
        errors.append('subtype {!r} is not allowed for family {} (allowed: {})'.format(
            label.subtype, label.family.value, ', '.join(allowed)))

    for key, value in label.attributes.items():
        if key not in ATTRIBUTE_VOCABULARY:
            errors.append('unknown attribute {!r} (known: {})'.format(
                key, ', '.join(sorted(ATTRIBUTE_VOCABULARY))))
        elif value not in ATTRIBUTE_VOCABULARY[key]:
            errors.append('value {!r} is not allowed for attribute {!r} (allowed: {})'.format(
                value, key, ', '.join(ATTRIBUTE_VOCABULARY[key])))
    return errors


#: The vocabulary of neume types. `BasicNeumeType` already models the Gregorian neume names,
#: so the discovery workflow reuses it instead of inventing a second one.
NEUME_TYPE_VOCABULARY: List[str] = [UNKNOWN] + [t.name.lower() for t in BasicNeumeType]


def validate_neume_type(neume_type: str) -> List[str]:
    if neume_type not in NEUME_TYPE_VOCABULARY:
        return ['neume type {!r} is not allowed (allowed: {})'.format(
            neume_type, ', '.join(NEUME_TYPE_VOCABULARY))]
    return []


# ----------------------------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------------------------
@dataclass
class SymbolCandidate(DataClassJSONMixin):
    """One discovered symbol candidate.

    Geometry (`box`, `mask_rle`, `center_*`) is in height-normalised page coordinates so that
    a candidate is independent of the image resolution it was found on.
    """
    id: str
    run_id: str
    book: str
    page: str
    block_id: str
    line_id: str
    box: Box
    center_x: float
    center_y: float
    mask_rle: Optional[MaskRle] = None
    embedding_index: Optional[int] = None
    discovery_score: float = 0.0
    discovery_method: str = ''
    #: dotted path of similarity clusters, '-1' is the preserved outlier label. Refining
    #: cluster '17' yields '17.0', '17.1', ... and '17.-1'.
    cluster_id: str = '-1'
    review_state: ReviewState = ReviewState.UNREVIEWED
    label: SymbolLabel = field(default_factory=SymbolLabel)
    rejection_reason: Optional[RejectionReason] = None
    neume_id: Optional[str] = None
    #: field paths a human set explicitly, e.g. ['label.family', 'label.attributes.connection'].
    #: A bulk operation skips exactly these, so a per-instance correction survives it.
    manual_fields: List[str] = field(default_factory=list)
    #: '<mtime_ns of pcgts.json>:<line id>' -- lets a later run notice that the staff geometry
    #: a candidate was found on has changed in the meantime.
    source_geometry_version: str = ''
    created_at: str = ''
    updated_at: str = ''


@dataclass
class SymbolRelation(DataClassJSONMixin):
    """The link between two consecutive components of a neume."""
    from_id: str
    to_id: str
    kind: str  # 'gaped' | 'looped'
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class NeumeCandidate(DataClassJSONMixin):
    id: str
    grouping_run_id: str
    symbols_run_id: str
    book: str
    page: str
    line_id: str
    box: Box
    component_ids: List[str] = field(default_factory=list)  # ordered by component centre x
    relations: List[SymbolRelation] = field(default_factory=list)
    mask_rle: Optional[MaskRle] = None
    embedding_index: Optional[int] = None
    grouping_score: float = 0.0
    cluster_id: str = '-1'
    grouping_state: GroupingState = GroupingState.PROPOSED
    #: independent of `grouping_state`: a CONFIRMED group whose type is still UNKNOWN is a
    #: perfectly valid state and must never be treated as an invalid group.
    neume_type: str = UNKNOWN
    attributes: Dict[str, str] = field(default_factory=dict)
    manual_fields: List[str] = field(default_factory=list)
    created_at: str = ''
    updated_at: str = ''
