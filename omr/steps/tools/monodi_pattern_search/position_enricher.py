"""
Enrich monodi+ transcriptions with position data from ommr4all PCGTS files.

Background
----------
The monodi+ format carries richer symbol vocabulary (liquescent, oriscus,
strophicus, …) but no positional data.  The ommr4all PCGTS format records
precise height-normalised pixel coordinates and staff positions for every
note, but uses a simpler vocabulary.

This module aligns both note sequences and injects PCGTS position data into
the monodi JSON structure.

Alignment strategy
------------------
Two modes are supported, controlled by ``use_syllables``:

Pure-pitch alignment  (``use_syllables=False``)
    A single edlib global alignment on pitch keys ("A"–"G").  Used as a
    fallback when no syllable annotations are available.

Syllable-guided two-level alignment  (``use_syllables=True``, default)
    **Level 1 – syllable alignment**
        The syllable-text sequences from both sides are aligned with edlib.
        Each PCGTS note is mapped to its syllable text via the PCGTS
        ``annotations.syllableConnectors`` (``noteID → syllableID →
        syllable.text``).  The monodi side already has syllable texts
        attached to each note directly.

        Exact matching on *normalised* syllable texts (lower-case, hyphens
        stripped) is used.  Only '=' (exact-match) spans in the CIGAR are
        treated as anchored syllable pairs; mismatches are ignored so that a
        mistyped syllable in one source does not corrupt an entire section.

    **Level 2 – note alignment within matched syllables**
        For each anchored syllable pair, the contained notes are aligned
        with edlib on single-character pitch keys.  '=' spans are direct
        matches; positions between two matched notes are linearly
        interpolated (``matched=False``).

    When PCGTS annotations are absent (no ``syllableConnectors``), the
    enricher automatically falls back to pure-pitch alignment.

All alignment uses edlib in global (NW) mode with ``task='path'`` so that
the full CIGAR is returned.

Injected JSON format
--------------------
For every matched/interpolated monodi note a ``"position"`` key is added::

    {
        "uuid": "...",
        "base": "G",
        "octave": 4,
        "noteType": "Liquescent",
        "liquescent": true,
        "focus": false,
        "position": {
            "x":                 0.1234,
            "y":                 0.4567,
            "position_in_staff": 5,
            "line_id":           "<uuid>",
            "line_coords":       "x1,y1 x2,y2 ...",
            "region_id":         "<uuid>",
            "region_coords":     "x1,y1 ...",
            "matched":           true
        }
    }

``matched=True``  – directly aligned note pair.
``matched=False`` – x/y linearly interpolated between two matched neighbours.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import edlib

from database.file_formats.importer.mondodi.monodi_parser import (
    MonodiChant,
    MonodiNote,
    MonodiPageLine,
    MonodiSyllable,
)


# ---------------------------------------------------------------------------
# CIGAR helpers
# ---------------------------------------------------------------------------

_CIGAR_RE = re.compile(r"(\d+)([=XID])")


def _parse_cigar(cigar: str) -> List[Tuple[int, str]]:
    """Return list of (count, op) from an extended CIGAR string."""
    return [(int(n), op) for n, op in _CIGAR_RE.findall(cigar)]


def _cigar_to_pairs(
    cigar: str,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Convert CIGAR to (query_idx, target_idx) pairs.
    None on the query side means deletion (only in target);
    None on the target side means insertion (only in query).
    """
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    qi = ti = 0
    for count, op in _parse_cigar(cigar):
        for _ in range(count):
            if op in ("=", "X"):      # match or mismatch
                pairs.append((qi, ti))
                qi += 1
                ti += 1
            elif op == "I":           # insertion: query has it, target does not
                pairs.append((qi, None))
                qi += 1
            elif op == "D":           # deletion: target has it, query does not
                pairs.append((None, ti))
                ti += 1
    return pairs


# ---------------------------------------------------------------------------
# Pitch-key helpers
# ---------------------------------------------------------------------------

_VALID_BASES = set("ABCDEFG")
# NoteName enum: A=0, B=1, C=2, D=3, E=4, F=5, G=6  → index-to-letter in that order
_IDX_TO_LETTER = ["A", "B", "C", "D", "E", "F", "G"]


def _monodi_pitch_key(note: MonodiNote) -> str:
    # Normalise to uppercase: monodi JSON sometimes stores 'a'–'g' in lowercase.
    base = note.base.upper() if note.base else "?"
    return base if base in _VALID_BASES else "?"


def _pcgts_note_pitch_key(note_json: dict) -> str:
    """Single-char pitch for a PCGTS note JSON node (``pname`` int → A-G).

    The ``pname`` integer encodes ``NoteName`` enum values: A=0 … G=6.
    ``_IDX_TO_LETTER`` maps those indices to the corresponding letter.
    """
    try:
        return _IDX_TO_LETTER[int(note_json["pname"])]
    except (KeyError, IndexError, ValueError, TypeError):
        return "?"


# ---------------------------------------------------------------------------
# Syllable-text normalisation
# ---------------------------------------------------------------------------

def _normalise_syllable(text: str) -> str:
    """
    Normalise a syllable text for comparison.
    Lower-cased; leading/trailing hyphens and whitespace stripped.
    """
    return text.strip().strip("-").lower()


# ---------------------------------------------------------------------------
# PCGTS JSON extraction helpers
# ---------------------------------------------------------------------------

def _collect_pcgts_music_lines(page_json: dict) -> List[dict]:
    """
    Return music-line JSON dicts from a PCGTS page, sorted top→bottom by
    average coord y.  Each dict is augmented with ``_region_id`` and
    ``_region_coords``.
    """
    lines: List[dict] = []
    for block in page_json.get("blocks", []):
        if block.get("type") != "music":
            continue
        for line in block.get("lines", []):
            line = dict(line)   # shallow copy so we can add keys
            line["_region_id"]     = block.get("id", "")
            line["_region_coords"] = block.get("coords", "")
            lines.append(line)

    def _avg_y(line: dict) -> float:
        ys = []
        for pt in line.get("coords", "").strip().split():
            parts = pt.split(",")
            if len(parts) == 2:
                try:
                    ys.append(float(parts[1]))
                except ValueError:
                    pass
        return sum(ys) / len(ys) if ys else 0.0

    lines.sort(key=_avg_y)
    return lines


def _collect_pcgts_notes(music_line_json: dict) -> List[dict]:
    """All note-type symbol JSON nodes from a PCGTS music-line."""
    return [s for s in music_line_json.get("symbols", []) if s.get("type") == "note"]


def _pitch_overlap_score(
    monodi_notes: List[MonodiNote],
    pcgts_notes:  List[dict],
) -> int:
    """
    Fast heuristic: count how many distinct pitch letters appear in BOTH note
    lists.  Used to rank candidate PCGTS lines when the expected line index
    is out of range or empty.
    """
    if not monodi_notes or not pcgts_notes:
        return 0
    m_set = {_monodi_pitch_key(n) for n in monodi_notes} - {"?"}
    p_set = {_pcgts_note_pitch_key(n) for n in pcgts_notes} - {"?"}
    return len(m_set & p_set)


def _extract_pcgts_syllable_map(page_json: dict) -> Dict[str, str]:
    """
    Build a ``note_id → normalised_syllable_text`` mapping from PCGTS JSON.

    Reads:
    * ``page.blocks[type=lyrics].lines[].sentence.syllables[]``
      to build ``syllable_id → normalised_text``
    * ``page.annotations.connections[].syllableConnectors[]``
      to build ``note_id → syllable_id``

    Returns an empty dict if no annotation data is present.
    """
    # 1) Build syllable_id → normalised text
    syllable_texts: Dict[str, str] = {}
    for block in page_json.get("blocks", []):
        if block.get("type") not in ("lyrics", "paragraph", "heading"):
            continue
        for line in block.get("lines", []):
            for syl in line.get("sentence", {}).get("syllables", []):
                syl_id = syl.get("id", "")
                if syl_id:
                    syllable_texts[syl_id] = _normalise_syllable(syl.get("text", ""))

    # 2) Build note_id → syllable_text via connectors
    note_to_syllable: Dict[str, str] = {}
    for conn in page_json.get("annotations", {}).get("connections", []):
        for sc in conn.get("syllableConnectors", []):
            note_id = sc.get("noteID", "")
            syl_id  = sc.get("syllableID", "")
            if note_id and syl_id and syl_id in syllable_texts:
                note_to_syllable[note_id] = syllable_texts[syl_id]

    return note_to_syllable


# ---------------------------------------------------------------------------
# Typed-page adapter  (DatabaseBook / DatabasePage path)
# ---------------------------------------------------------------------------

def _adapt_typed_page(page) -> Tuple[List[dict], Dict[str, str]]:
    """
    Convert a typed PCGTS ``Page`` object to the intermediate dict format
    expected by the alignment functions.

    ``page.update_note_names()`` is called on the ``Page`` constructor, so
    pitch names are already resolved from clef context when this function
    runs.

    Parameters
    ----------
    page:
        A ``database.file_formats.pcgts.page.Page`` instance.

    Returns
    -------
    music_lines:
        Sorted list of music-line dicts (same schema as produced by
        :func:`_collect_pcgts_music_lines`).
    note_to_syllable:
        ``note_id → normalised_syllable_text`` mapping built from
        ``page.annotations.connections`` (empty dict if none present).
    """
    from database.file_formats.pcgts.page.musicsymbol import SymbolType, NoteName
    from database.file_formats.pcgts.page.definitions import BlockType

    # ---- Build music-line dicts ----------------------------------------
    music_lines: List[dict] = []
    for block in page.blocks:
        if block.block_type != BlockType.MUSIC:
            continue
        block_coords_str: str = block.coords.to_string() if block.coords is not None else ""
        for line in block.lines:
            note_dicts: List[dict] = []
            for sym in line.symbols:
                if sym.symbol_type != SymbolType.NOTE:
                    continue
                if sym.note_name == NoteName.UNDEFINED:
                    continue
                coord_str = f"{sym.coord.x},{sym.coord.y}" if sym.coord is not None else "0,0"
                note_dicts.append({
                    "type":            "note",
                    "id":              sym.id,
                    "coord":           coord_str,
                    "positionInStaff": sym.position_in_staff.value,
                    # pname: NoteName int value (A=0, B=1, C=2, D=3, E=4, F=5, G=6)
                    # _IDX_TO_LETTER[pname] gives the correct letter.
                    "pname":           sym.note_name.value,
                })
            line_coords_str: str = line.coords.to_string() if line.coords is not None else ""
            music_lines.append({
                "id":              line.id,
                "coords":          line_coords_str,
                "symbols":         note_dicts,
                "_region_id":      block.id,
                "_region_coords":  block_coords_str,
            })

    # Sort by average y (top → bottom)
    def _avg_y_typed(ml: dict) -> float:
        ys = []
        for pt in ml["coords"].strip().split():
            parts = pt.split(",")
            if len(parts) == 2:
                try:
                    ys.append(float(parts[1]))
                except ValueError:
                    pass
        return sum(ys) / len(ys) if ys else 0.0

    music_lines.sort(key=_avg_y_typed)

    # ---- Build note_id → syllable text from annotations ----------------
    note_to_syllable: Dict[str, str] = {}
    for conn in page.annotations.connections:
        for sc in conn.syllable_connections:
            note_id  = sc.note.id
            syl_text = _normalise_syllable(sc.syllable.text)
            note_to_syllable[note_id] = syl_text

    return music_lines, note_to_syllable


# ---------------------------------------------------------------------------
# Position data container
# ---------------------------------------------------------------------------

@dataclass
class InjectedPosition:
    """Position data copied from one PCGTS note to one monodi note."""
    x: float
    y: float
    position_in_staff: int
    line_id: str
    line_coords: str
    region_id: str
    region_coords: str
    matched: bool            # True = direct alignment; False = interpolated
    pcgts_symbol_id: str = ""  # ID of the matched PCGTS note; empty for interpolated

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "x":                 self.x,
            "y":                 self.y,
            "position_in_staff": self.position_in_staff,
            "line_id":           self.line_id,
            "line_coords":       self.line_coords,
            "region_id":         self.region_id,
            "region_coords":     self.region_coords,
            "matched":           self.matched,
        }
        if self.pcgts_symbol_id:
            d["pcgts_symbol_id"] = self.pcgts_symbol_id
        return d


def _make_position(
    pcgts_sym: dict,
    pcgts_line: dict,
    matched: bool,
) -> InjectedPosition:
    coord_str = pcgts_sym.get("coord", "0,0")
    parts = coord_str.split(",")
    return InjectedPosition(
        x=float(parts[0]) if len(parts) > 0 else 0.0,
        y=float(parts[1]) if len(parts) > 1 else 0.0,
        position_in_staff=int(pcgts_sym.get("positionInStaff", -1000)),
        line_id=pcgts_line.get("id", ""),
        line_coords=pcgts_line.get("coords", ""),
        region_id=pcgts_line.get("_region_id", ""),
        region_coords=pcgts_line.get("_region_coords", ""),
        matched=matched,
        pcgts_symbol_id=pcgts_sym.get("id", "") if matched else "",
    )


# ---------------------------------------------------------------------------
# Low-level: align two note lists by pitch with edlib
# ---------------------------------------------------------------------------

def _align_notes_edlib(
    monodi_notes:    List[MonodiNote],
    pcgts_notes:     List[dict],
    pcgts_line:      dict,
    accept_mismatch: bool = False,
) -> Dict[str, InjectedPosition]:
    """
    Align *monodi_notes* to *pcgts_notes* on pitch keys using edlib.
    Returns monodi UUID → InjectedPosition for matched and interpolated notes.

    Parameters
    ----------
    accept_mismatch:
        When *True*, edlib ``X`` (mismatch) pairs are accepted in addition to
        ``=`` (exact) pairs.  Use this when a higher-level anchor (e.g. a
        matching syllable text) already provides strong evidence that the two
        notes correspond, so a single-step pitch discrepancy should not block
        the alignment.  Pure-pitch fallback calls should leave this *False*.
    """
    if not monodi_notes or not pcgts_notes:
        return {}

    monodi_keys = "".join(_monodi_pitch_key(n) for n in monodi_notes)
    pcgts_keys  = "".join(_pcgts_note_pitch_key(s) for s in pcgts_notes)

    result = edlib.align(monodi_keys, pcgts_keys, mode="NW", task="path")
    pairs  = _cigar_to_pairs(result["cigar"])

    # monodi index → pcgts index
    direct: Dict[int, int] = {}
    for qi, ti in pairs:
        if qi is not None and ti is not None:
            if monodi_keys[qi] == pcgts_keys[ti]:
                direct[qi] = ti          # exact pitch match — always accept
            elif accept_mismatch:
                direct[qi] = ti          # mismatch tolerated inside syllable pair

    return _build_positions(monodi_notes, pcgts_notes, pcgts_line, direct)


def _build_positions(
    monodi_notes: List[MonodiNote],
    pcgts_notes:  List[dict],
    pcgts_line:   dict,
    direct: Dict[int, int],
) -> Dict[str, InjectedPosition]:
    """
    Given a dict of direct monodi_idx → pcgts_idx matches, build
    InjectedPosition entries and linearly interpolate unmatched interiors.
    """
    result: Dict[str, InjectedPosition] = {}

    # Direct matches
    for ai, bi in direct.items():
        result[monodi_notes[ai].uuid] = _make_position(pcgts_notes[bi], pcgts_line, matched=True)

    # Interpolate gaps between consecutive direct matches
    matched_indices = sorted(direct.keys())
    for k in range(len(matched_indices) - 1):
        left_ai  = matched_indices[k]
        right_ai = matched_indices[k + 1]
        gap = list(range(left_ai + 1, right_ai))
        if not gap:
            continue
        left_pos  = result[monodi_notes[left_ai].uuid]
        right_pos = result[monodi_notes[right_ai].uuid]
        n_steps   = len(gap) + 1
        for step, ai in enumerate(gap, start=1):
            t = step / n_steps
            result[monodi_notes[ai].uuid] = InjectedPosition(
                x=left_pos.x + t * (right_pos.x - left_pos.x),
                y=left_pos.y + t * (right_pos.y - left_pos.y),
                position_in_staff=(
                    left_pos.position_in_staff if t <= 0.5 else right_pos.position_in_staff
                ),
                line_id=pcgts_line.get("id", ""),
                line_coords=pcgts_line.get("coords", ""),
                region_id=pcgts_line.get("_region_id", ""),
                region_coords=pcgts_line.get("_region_coords", ""),
                matched=False,
            )

    return result


# ---------------------------------------------------------------------------
# Syllable-guided two-level alignment
# ---------------------------------------------------------------------------

def _group_pcgts_notes_by_syllable(
    pcgts_notes: List[dict],
    note_to_syllable: Dict[str, str],
) -> Tuple[List[str], List[List[dict]]]:
    """
    Group PCGTS notes by syllable text in document order.

    Orphaned notes — those with no syllable connector AND that appear before
    any connector has been seen (e.g. a neume that carries over from the
    previous staff line) — are buffered and prepended to the first real
    syllable group.  This prevents them from forming a silent empty-text
    group that the level-1 syllable alignment can never match.

    For orphaned notes that appear *after* at least one connector has been
    seen (typical neume continuation notes), fill-forward assigns them to
    the current syllable group as before.

    Returns:
        syllable_texts  – list of normalised syllable texts (one per group)
        note_groups     – parallel list of note lists
    """
    syllable_texts: List[str] = []
    note_groups: List[List[dict]] = []
    current_syl: Optional[str] = None
    current_notes: List[dict] = []
    # Notes seen before the very first syllable connector on this line.
    leading_orphans: List[dict] = []

    for note in pcgts_notes:
        syl_text = note_to_syllable.get(note.get("id", ""), "")

        if not syl_text:
            if current_syl is not None:
                # Fill-forward: neume-continuation note → same syllable group.
                syl_text = current_syl
            else:
                # Before any connector has been seen: buffer for prepending to
                # the first real syllable group instead of forming an orphan group.
                leading_orphans.append(note)
                continue

        if syl_text != current_syl:
            if current_notes:
                syllable_texts.append(current_syl or "")
                note_groups.append(current_notes)
            current_syl   = syl_text
            # Prepend any buffered leading orphans to the first real group.
            current_notes = leading_orphans + [note]
            leading_orphans = []
        else:
            current_notes.append(note)

    if current_notes:
        syllable_texts.append(current_syl or "")
        note_groups.append(current_notes)

    return syllable_texts, note_groups


def _align_syllables_edlib(
    monodi_texts: List[str],
    pcgts_texts:  List[str],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Align two lists of normalised syllable-text tokens with edlib.

    edlib supports iterables of hashable objects so the text strings are
    passed directly as tokens (the 256-unique-value limit applies — we rely
    on the texts fitting within that; for very long chants a hash could be
    used but in practice chant syllables are limited).

    Returns (query_idx, target_idx) pairs from the CIGAR.
    """
    if not monodi_texts or not pcgts_texts:
        return []

    # If combined unique count exceeds 255, fall back to character-level
    # alignment of concatenated texts (less precise but safe).
    combined = set(monodi_texts) | set(pcgts_texts)
    if len(combined) > 255:
        # Encode as integers 0-254 via a truncated hash; collisions mean
        # occasional false matches but this is a very rare edge case.
        vocab = {t: i % 255 for i, t in enumerate(sorted(combined))}
        monodi_int = [vocab[t] for t in monodi_texts]
        pcgts_int  = [vocab[t] for t in pcgts_texts]
        result = edlib.align(monodi_int, pcgts_int, mode="NW", task="path")
    else:
        result = edlib.align(monodi_texts, pcgts_texts, mode="NW", task="path")

    return _cigar_to_pairs(result["cigar"])


def _align_with_syllables(
    monodi_line:       MonodiPageLine,
    pcgts_notes:       List[dict],
    pcgts_line:        dict,
    note_to_syllable:  Dict[str, str],
    include_note_types: Optional[List[str]],
) -> Dict[str, InjectedPosition]:
    """
    Two-level alignment using syllables as anchors.

    Level 1: align syllable-text sequences with edlib.
    Level 2: for each matched syllable pair, align notes by pitch with edlib.
    """
    # --- Monodi side: syllables that contain notes ---
    monodi_syllables: List[MonodiSyllable] = [
        s for s in monodi_line.syllables if s.notes
    ]
    if include_note_types is not None:
        monodi_syllables = [
            s for s in monodi_syllables
            if any(n.note_type in include_note_types for n in s.notes)
        ]

    # --- PCGTS side: group notes by syllable ---
    pcgts_syl_texts, pcgts_note_groups = _group_pcgts_notes_by_syllable(
        pcgts_notes, note_to_syllable
    )

    # --- Level 1: syllable alignment ---
    monodi_syl_texts = [_normalise_syllable(s.text) for s in monodi_syllables]
    norm_pcgts_texts = [_normalise_syllable(t) for t in pcgts_syl_texts]

    syl_pairs = _align_syllables_edlib(monodi_syl_texts, norm_pcgts_texts)

    # Accept only exact-text matches as anchors
    all_positions: Dict[str, InjectedPosition] = {}

    for qi, ti in syl_pairs:
        if qi is None or ti is None:
            continue
        if monodi_syl_texts[qi] != norm_pcgts_texts[ti]:
            continue   # substitution — skip (texts differ)

        monodi_syl   = monodi_syllables[qi]
        pcgts_grp    = pcgts_note_groups[ti]

        # Filter monodi notes for this syllable
        m_notes = monodi_syl.notes
        if include_note_types is not None:
            m_notes = [n for n in m_notes if n.note_type in include_note_types]

        # Level 2: note alignment within the syllable pair.
        # accept_mismatch=True: the syllable-text match already anchors
        # both notes to the same syllable, so a small pitch discrepancy
        # should not prevent the position from being assigned.
        positions = _align_notes_edlib(
            m_notes, pcgts_grp, pcgts_line, accept_mismatch=True
        )
        all_positions.update(positions)

    return all_positions


# ---------------------------------------------------------------------------
# JSON enrichment
# ---------------------------------------------------------------------------

def _inject_into_data_json(
    data: dict,
    positions_by_uuid: Dict[str, InjectedPosition],
) -> dict:
    """Return a deep copy of *data* with ``"position"`` keys injected."""
    data = copy.deepcopy(data)

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("kind") == "Syllable":
            for spaced in node.get("notes", {}).get("spaced", []):
                for non_spaced in spaced.get("nonSpaced", []):
                    for grouped in non_spaced.get("grouped", []):
                        uuid = grouped.get("uuid", "")
                        if uuid in positions_by_uuid:
                            grouped["position"] = positions_by_uuid[uuid].to_dict()
        for child in node.get("children", []):
            _walk(child)

    _walk(data)
    return data


# ---------------------------------------------------------------------------
# High-level enricher
# ---------------------------------------------------------------------------

class MonodiPositionEnricher:
    """
    Align monodi notes to PCGTS notes and inject position data.

    Parameters
    ----------
    use_syllables:
        If *True* (default), use the PCGTS syllable annotations for a
        two-level alignment (syllables first, then notes within syllables).
        Falls back to pure-pitch alignment when no annotation data is found.
    include_note_types:
        Restrict which monodi note types participate in the alignment.
        ``None`` (default) includes all types.
    """

    def __init__(
        self,
        use_syllables: bool = True,
        include_note_types: Optional[List[str]] = None,
    ) -> None:
        self.use_syllables      = use_syllables
        self.include_note_types = include_note_types

    def _filter(self, notes: List[MonodiNote]) -> List[MonodiNote]:
        if self.include_note_types is None:
            return notes
        return [n for n in notes if n.note_type in self.include_note_types]

    def _align_line(
        self,
        monodi_line:      MonodiPageLine,
        pcgts_line:       dict,
        note_to_syllable: Dict[str, str],
    ) -> Dict[str, InjectedPosition]:
        """Align one monodi page-line against one PCGTS music line."""
        pcgts_notes = _collect_pcgts_notes(pcgts_line)

        if self.use_syllables and note_to_syllable:
            positions = _align_with_syllables(
                monodi_line, pcgts_notes, pcgts_line,
                note_to_syllable, self.include_note_types,
            )
            # Fall back to pure pitch for any monodi note still without a position.
            # Exclude PCGTS notes that were already directly matched so the
            # fallback cannot assign the same PCGTS note to a second monodi note.
            unpositioned = [
                n for n in self._filter(monodi_line.all_notes)
                if n.uuid not in positions
            ]
            if unpositioned:
                matched_pcgts_ids = {
                    pos.pcgts_symbol_id
                    for pos in positions.values()
                    if pos.matched and pos.pcgts_symbol_id
                }
                pcgts_for_fallback = [
                    n for n in pcgts_notes
                    if n.get("id", "") not in matched_pcgts_ids
                ]
                fallback = _align_notes_edlib(unpositioned, pcgts_for_fallback, pcgts_line)
                for uuid, pos in fallback.items():
                    if uuid not in positions:
                        positions[uuid] = pos
            return positions

        # Pure-pitch fallback
        return _align_notes_edlib(
            self._filter(monodi_line.all_notes), pcgts_notes, pcgts_line
        )

    def _resolve_pcgts_line_idx(
        self,
        expected_idx:  int,
        pcgts_lines:   List[dict],
        monodi_notes:  List[MonodiNote],
        search_radius: int,
    ) -> Optional[int]:
        """
        Return the best PCGTS-line index to use for *monodi_notes*.

        Strategy
        --------
        1. If *expected_idx* is in range and the line has notes → use it directly.
        2. If *expected_idx* is in range but the line has **no notes** (empty or
           not yet transcribed staff) → search within ``±search_radius`` for the
           nearest line that does have notes.
        3. If *expected_idx* is **out of range** (PCGTS page shorter than the
           monodi line-numbering expects) → search the entire reachable window
           ``[expected_idx - search_radius, len(pcgts_lines) - 1]`` and rank
           candidates by pitch-key overlap with the monodi notes so that the
           line whose note content best matches is preferred.

        Returns *None* when no usable candidate is found.
        """
        if not pcgts_lines:
            return None

        # --- Case 1: expected index in range and has notes ---
        if 0 <= expected_idx < len(pcgts_lines):
            if _collect_pcgts_notes(pcgts_lines[expected_idx]):
                return expected_idx

            # --- Case 2: expected index in range but line is empty ---
            for delta in range(1, search_radius + 1):
                for sign in (-1, 1):
                    idx = expected_idx + sign * delta
                    if 0 <= idx < len(pcgts_lines):
                        if _collect_pcgts_notes(pcgts_lines[idx]):
                            return idx
            # All nearby lines also empty → return expected as-is (alignment
            # will simply produce no positions but at least no line is skipped).
            return expected_idx

        # --- Case 3: expected index out of range ---
        # Build a ranked list of candidates within the reachable window.
        lo = max(0, expected_idx - search_radius)
        hi = len(pcgts_lines) - 1
        if lo > hi:
            return None

        best_idx   = hi          # default: last line
        best_score = -1
        for idx in range(lo, hi + 1):
            notes = _collect_pcgts_notes(pcgts_lines[idx])
            if not notes:
                continue
            score = _pitch_overlap_score(monodi_notes, notes)
            if score > best_score:
                best_score = score
                best_idx   = idx

        return best_idx

    def enrich_full_chant(
        self,
        chant:               MonodiChant,
        data_json:           dict,
        folio_to_pcgts_json: Optional[Dict[str, dict]] = None,
        folio_to_db_page:    Optional[Dict[str, Any]]  = None,
        search_radius:       int = 1,
    ) -> dict:
        """
        Enrich all folios of a chant in one call.

        Parameters
        ----------
        chant:
            Parsed chant (``page_lines`` must be populated).
        data_json:
            Raw ``data.json`` dict — a deep copy is created before injection.
        folio_to_db_page:
            ``{folio_id: DatabasePage}`` — preferred.  When provided, PCGTS is
            loaded via the typed ``DatabasePage.pcgts().page`` API.  Pitch
            names are resolved from clef context via ``update_note_names()``.
        folio_to_pcgts_json:
            ``{folio_id: pcgts_json}`` — fallback (raw JSON dicts).  Used when
            ``folio_to_db_page`` is *None*.
        search_radius:
            How many PCGTS lines to search around the expected index when the
            expected line is out of range or has no notes.  Default ``1``.
            Increase to ``2`` or ``3`` when PCGTS pages are missing several
            staves.  Set to ``0`` to restore the previous strict behaviour.
        """
        import logging
        logger = logging.getLogger(__name__)

        if folio_to_db_page is None and folio_to_pcgts_json is None:
            raise ValueError("Provide either folio_to_db_page or folio_to_pcgts_json.")

        # Pre-compute PCGTS music lines and syllable maps per folio
        pcgts_lines_by_folio: Dict[str, List[dict]] = {}
        syl_map_by_folio:     Dict[str, Dict[str, str]] = {}

        if folio_to_db_page is not None:
            # --- Typed DatabasePage path (preferred) ---
            for folio, db_page in folio_to_db_page.items():
                page = db_page.pcgts().page          # triggers update_note_names via from_json
                music_lines, note_to_syl = _adapt_typed_page(page)
                pcgts_lines_by_folio[folio] = music_lines
                syl_map_by_folio[folio]     = note_to_syl if self.use_syllables else {}
        else:
            # --- Raw JSON fallback path ---
            for folio, pcgts_json in folio_to_pcgts_json.items():
                page_node = pcgts_json.get("page", pcgts_json)
                pcgts_lines_by_folio[folio] = _collect_pcgts_music_lines(page_node)
                syl_map_by_folio[folio]     = (
                    _extract_pcgts_syllable_map(page_node)
                    if self.use_syllables
                    else {}
                )

        all_positions: Dict[str, InjectedPosition] = {}

        for monodi_line in chant.page_lines:
            folio = str(monodi_line.folio)

            if folio not in pcgts_lines_by_folio:
                logger.warning(
                    "Folio '%s' has no PCGTS data — %d notes unpositioned",
                    folio, len(monodi_line.all_notes),
                )
                continue

            pcgts_lines  = pcgts_lines_by_folio[folio]
            # line_number is 1-based within the folio; convert to 0-based index.
            # zeilenstart in the chant meta gives the starting line number so
            # chants that begin mid-page correctly index into the PCGTS lines.
            expected_idx = monodi_line.line_number - 1

            line_idx = self._resolve_pcgts_line_idx(
                expected_idx, pcgts_lines,
                self._filter(monodi_line.all_notes),
                search_radius,
            )

            if line_idx is None:
                logger.warning(
                    "Folio '%s' line %d (expected pcgts idx %d): "
                    "PCGTS page has %d lines — no candidate found, %d notes unpositioned",
                    folio, monodi_line.line_number, expected_idx,
                    len(pcgts_lines), len(monodi_line.all_notes),
                )
                continue

            if line_idx != expected_idx:
                logger.debug(
                    "Folio '%s' line %d: expected pcgts idx %d, "
                    "using idx %d (page has %d lines)",
                    folio, monodi_line.line_number, expected_idx,
                    line_idx, len(pcgts_lines),
                )

            positions = self._align_line(
                monodi_line,
                pcgts_lines[line_idx],
                syl_map_by_folio[folio],
            )
            all_positions.update(positions)

        return _inject_into_data_json(data_json, all_positions)

    def enrich(
        self,
        chant:           MonodiChant,
        folio:           str,
        pcgts_page_json: Optional[dict] = None,
        db_page:         Optional[Any]  = None,
        data_json:       Optional[dict] = None,
        data_json_path:  Optional[str]  = None,
    ) -> dict:
        """
        Enrich one folio of a chant (convenience wrapper for single pages).

        Pass either *db_page* (``DatabasePage``, preferred) or
        *pcgts_page_json* (raw dict, fallback).
        """
        if data_json is None:
            if data_json_path is None:
                raise ValueError("Provide either data_json or data_json_path.")
            with open(data_json_path, "r", encoding="utf-8") as fh:
                data_json = json.load(fh)

        if db_page is not None:
            return self.enrich_full_chant(
                chant, data_json, folio_to_db_page={folio: db_page}
            )
        if pcgts_page_json is not None:
            return self.enrich_full_chant(
                chant, data_json, folio_to_pcgts_json={folio: pcgts_page_json}
            )
        raise ValueError("Provide either db_page or pcgts_page_json.")


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def alignment_stats(enriched_data: dict) -> Dict[str, int]:
    """Count matched, interpolated, and unpositioned notes in enriched JSON."""
    stats = {"total": 0, "matched": 0, "interpolated": 0, "unpositioned": 0}

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("kind") == "Syllable":
            for spaced in node.get("notes", {}).get("spaced", []):
                for non_spaced in spaced.get("nonSpaced", []):
                    for grouped in non_spaced.get("grouped", []):
                        stats["total"] += 1
                        pos = grouped.get("position")
                        if pos is None:
                            stats["unpositioned"] += 1
                        elif pos.get("matched"):
                            stats["matched"] += 1
                        else:
                            stats["interpolated"] += 1
        for child in node.get("children", []):
            _walk(child)

    _walk(enriched_data)
    return stats
