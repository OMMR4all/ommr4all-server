"""
Extended monodi+ importer for pattern search.

Parses a manuscript export folder (one meta.json + N chant sub-folders, each
with its own meta.json and data.json) into a structured in-memory representation
that is suitable for melodic pattern matching.

Graphical-connection semantics (matching the encoding in simple_import.py):
  spaced  → NEUME_START for the first note of every spaced group
  nonSpaced inside a spaced → GAPED for the first note of every subsequent group
  grouped inside a nonSpaced → LOOPED for every note after the first
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Graphical-connection constants  (match GraphicalConnectionType enum order)
# ---------------------------------------------------------------------------
CONN_NEUME_START = 0   # first note of a new neume (spaced group)
CONN_GAPED       = 1   # start of a new nonSpaced sub-group within a neume
CONN_LOOPED      = 2   # ligature — subsequent note inside a grouped cluster

# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------
_NOTE_DIATONIC: Dict[str, int] = {"C": 0, "D": 1, "E": 2, "F": 3,
                                   "G": 4, "A": 5, "B": 6}
_NOTE_SEMITONE: Dict[str, int] = {"C": 0, "D": 2, "E": 4, "F": 5,
                                   "G": 7, "A": 9, "B": 11}
_ALTERATION_SEMITONE: Dict[str, int] = {"Flat": -1, "Sharp": 1, "Natural": 0}



@dataclass
class MonodiNote:
    uuid: str
    base: str        # "A" … "G"
    octave: int
    note_type: str   # "Normal" | "Liquescent" | "Oriscus" | "Strophicus" |
                     # "Flat" | "Sharp" | "Natural" | "Ascending" | "Descending"
    liquescent: bool
    graphical_connection: int

    @property
    def abs_pitch_diatonic(self) -> int:
        """Absolute diatonic pitch: octave*7 + step (C=0 … B=6)."""
        return self.octave * 7 + _NOTE_DIATONIC.get(self.base, 0)

    @property
    def abs_pitch_semitone(self) -> int:
        """Absolute semitone pitch with accidental correction."""
        base = self.octave * 12 + _NOTE_SEMITONE.get(self.base, 0)
        return base + _ALTERATION_SEMITONE.get(self.note_type, 0)


@dataclass
class MonodiSyllable:
    uuid: str
    text: str
    syllable_type: str   # "Normal" | "WithoutNotes" | "EditorialEllipsis" | "SourceEllipsis"
    notes: List[MonodiNote] = field(default_factory=list)


@dataclass
class MonodiRow:
    row_number: int           # 1-indexed within the chant

    # The syllable objects in order; some may have no notes.
    syllables: List[MonodiSyllable] = field(default_factory=list)

    @property
    def all_notes(self) -> List[MonodiNote]:
        return [note for syl in self.syllables for note in syl.notes]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.syllables if s.text.strip())


@dataclass
class ChantMeta:
    id: str
    quelle_id: str
    dokumenten_id: str
    festtag: str
    feier: str
    textinitium: str
    foliostart: str
    zeilenstart: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonodiPageLine:
    """
    One physical staff line on one folio page.

    Created by :func:`parse_data_json_by_page`, which walks the flat sequence
    of ``LineChange`` / ``FolioChange`` / ``Syllable`` nodes across all
    ``ZeileContainer`` blocks and splits them into individual manuscript lines.

    Attributes
    ----------
    folio:
        Folio identifier string as it appears in ``FolioChange.text`` or is
        taken from the chant's ``foliostart`` meta field for the very first
        line.  Example values: ``"18"``, ``"86v"``.
    line_number:
        1-indexed line number on *this* folio.  The first line of the chant
        starts at ``int(ChantMeta.zeilenstart)``; the counter resets to 1
        whenever a new folio begins.
    syllables:
        The syllables (and their notes) that belong to this line.
    """
    folio: str
    line_number: int
    syllables: List[MonodiSyllable] = field(default_factory=list)

    @property
    def all_notes(self) -> List[MonodiNote]:
        return [n for s in self.syllables for n in s.notes]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.syllables if s.text.strip())


@dataclass
class MonodiChant:
    meta: ChantMeta
    rows: List[MonodiRow]          # one entry per ZeileContainer (section view)
    page_lines: List[MonodiPageLine] = field(default_factory=list)  # one per physical staff line


@dataclass
class ManuscriptMeta:
    id: str
    quellensigle: str
    bibliothek: str
    bibliothekssignatur: str
    datierung: str
    herkunftsort: str


@dataclass
class MonodiManuscript:
    meta: ManuscriptMeta
    chants: List[MonodiChant]


def _extract_notes(note_dict: dict) -> List[MonodiNote]:
    """
    Parse the ``notes`` sub-dict of a Syllable node into an ordered list of
    MonodiNote objects with correct graphical-connection values.

    Hierarchy:  spaced → nonSpaced → grouped

    Connection assignment (mirrors simple_import.py logic):
      - First note of each ``spaced`` group          → CONN_NEUME_START
      - First note of each subsequent ``nonSpaced``  → CONN_GAPED
      - Every subsequent note inside ``grouped``     → CONN_LOOPED
    """
    notes: List[MonodiNote] = []
    conn_state = CONN_NEUME_START

    for spaced in note_dict.get("spaced", []):
        for non_spaced in spaced.get("nonSpaced", []):
            for grouped in non_spaced.get("grouped", []):
                notes.append(MonodiNote(
                    uuid=grouped.get("uuid", ""),
                    base=grouped.get("base", "C"),
                    octave=grouped.get("octave", 4),
                    note_type=grouped.get("noteType", "Normal"),
                    liquescent=grouped.get("liquescent", False),
                    graphical_connection=conn_state,
                ))
                conn_state = CONN_LOOPED   # subsequent notes in the same grouped
            conn_state = CONN_GAPED        # next nonSpaced in the same spaced
        conn_state = CONN_NEUME_START      # next spaced group

    return notes


def _collect_row_containers(node: dict, accumulator: List[dict]) -> None:
    """Depth-first search for ZeileContainer nodes."""
    if node.get("kind") == "ZeileContainer":
        accumulator.append(node)
        return  # do not recurse into a row's children for nested rows
    for child in node.get("children", []):
        _collect_row_containers(child, accumulator)


def parse_data_json(data: dict) -> List[MonodiRow]:
    """
    Parse a monodi+ ``data.json`` dict into an ordered list of :class:`MonodiRow`
    objects (one per ZeileContainer in document order).
    """
    row_containers: List[dict] = []
    _collect_row_containers(data, row_containers)

    rows: List[MonodiRow] = []
    for row_idx, container in enumerate(row_containers):
        syllables: List[MonodiSyllable] = []
        for child in container.get("children", []):
            if child.get("kind") == "Syllable":
                notes = _extract_notes(child.get("notes", {"spaced": []}))
                syllables.append(MonodiSyllable(
                    uuid=child.get("uuid", ""),
                    text=child.get("text", ""),
                    syllable_type=child.get("syllableType", "Normal"),
                    notes=notes,
                ))
        rows.append(MonodiRow(row_number=row_idx + 1, syllables=syllables))

    return rows


def parse_data_json_by_page(
    data: dict,
    foliostart: str,
    zeilenstart: int,
) -> List[MonodiPageLine]:
    """
    Parse a monodi+ ``data.json`` into an ordered list of :class:`MonodiPageLine`
    objects — one entry per physical staff line on the manuscript page.

    Algorithm
    ---------
    All ``ZeileContainer`` nodes are collected in document order.  Their
    children are then walked **as a single flat stream**.  Whenever a
    ``LineChange`` is encountered the current line segment is closed and a new
    one opened on the *same* folio (line number incremented).  Whenever a
    ``FolioChange`` is encountered the current segment is closed and the folio
    switches to the value in ``FolioChange.text``; the line counter resets to
    1 (first line of the new folio, to be incremented by the next
    ``LineChange`` if one immediately follows).

    The very first content (before any explicit ``LineChange``/``FolioChange``)
    is placed on folio *foliostart*, line *zeilenstart*.

    Parameters
    ----------
    data:
        Parsed ``data.json`` dict (``RootContainer``).
    foliostart:
        Folio identifier string from the chant's ``meta.json``
        (e.g. ``"18"`` or ``"86v"``).
    zeilenstart:
        Integer line number of the first line of this chant on *foliostart*.
    """
    row_containers: List[dict] = []
    _collect_row_containers(data, row_containers)

    page_lines: List[MonodiPageLine] = []
    current_folio = foliostart
    current_line  = zeilenstart
    current_syllables: List[MonodiSyllable] = []

    def _flush() -> None:
        """Save the current syllable buffer as a MonodiPageLine."""
        if current_syllables:
            page_lines.append(MonodiPageLine(
                folio=current_folio,
                line_number=current_line,
                syllables=list(current_syllables),
            ))
            current_syllables.clear()

    for container in row_containers:
        for child in container.get("children", []):
            kind = child.get("kind")

            if kind == "LineChange":
                _flush()
                current_line += 1

            elif kind == "FolioChange":
                _flush()
                current_folio = str(child.get("text", current_folio)).strip()
                current_line  = 1   # first line of the new folio

            elif kind == "Syllable":
                notes = _extract_notes(child.get("notes", {"spaced": []}))
                current_syllables.append(MonodiSyllable(
                    uuid=child.get("uuid", ""),
                    text=child.get("text", ""),
                    syllable_type=child.get("syllableType", "Normal"),
                    notes=notes,
                ))

    _flush()   # save any trailing content
    return page_lines


def parse_chant_meta(meta_data: dict) -> ChantMeta:
    return ChantMeta(
        id=meta_data.get("id", ""),
        quelle_id=meta_data.get("quelle_id", ""),
        dokumenten_id=meta_data.get("dokumenten_id", ""),
        festtag=meta_data.get("festtag", ""),
        feier=meta_data.get("feier", ""),
        textinitium=meta_data.get("textinitium", ""),
        foliostart=meta_data.get("foliostart", ""),
        zeilenstart=meta_data.get("zeilenstart", ""),
        additional_data=meta_data.get("additionalData", {}),
    )


def load_manuscript(folder_path: str) -> MonodiManuscript:
    """
    Load a complete monodi+ manuscript export from *folder_path*.

    Expected layout::

        <folder_path>/
            meta.json              ← manuscript-level metadata
            <chant-uuid>/
                meta.json          ← chant-level metadata
                data.json          ← transcription (RootContainer JSON)
            …

    Returns a :class:`MonodiManuscript` with all chants parsed.
    """
    meta_path = os.path.join(folder_path, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta_data = json.load(fh)

    man_meta = ManuscriptMeta(
        id=meta_data.get("id", os.path.basename(folder_path)),
        quellensigle=meta_data.get("quellensigle", ""),
        bibliothek=meta_data.get("bibliothek", ""),
        bibliothekssignatur=meta_data.get("bibliothekssignatur", ""),
        datierung=meta_data.get("datierung", ""),
        herkunftsort=meta_data.get("herkunftsort", ""),
    )

    chants: List[MonodiChant] = []
    for entry in sorted(os.scandir(folder_path), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        chant_meta_path = os.path.join(entry.path, "meta.json")
        chant_data_path = os.path.join(entry.path, "data.json")
        if not (os.path.exists(chant_meta_path) and os.path.exists(chant_data_path)):
            continue

        with open(chant_meta_path, "r", encoding="utf-8") as fh:
            chant_meta_data = json.load(fh)
        with open(chant_data_path, "r", encoding="utf-8") as fh:
            chant_data = json.load(fh)

        chant_meta = parse_chant_meta(chant_meta_data)
        try:
            zeilenstart = int(chant_meta.zeilenstart)
        except (ValueError, TypeError):
            zeilenstart = 1

        chants.append(MonodiChant(
            meta=chant_meta,
            rows=parse_data_json(chant_data),
            page_lines=parse_data_json_by_page(
                chant_data,
                foliostart=chant_meta.foliostart,
                zeilenstart=zeilenstart,
            ),
        ))

    return MonodiManuscript(meta=man_meta, chants=chants)
