"""
Alignment and comparison between a MonodiPageLine (from monodi+ export) and
the notes of a PCGTS music line (from ommr4all).

Overview
--------
Both sides carry a sequence of pitches (note name + octave).  We use
``difflib.SequenceMatcher`` to align them, exactly as the existing
``omr/steps/tools/predictor.py`` does.  This produces a list of
:class:`LineComparisonSegment` objects that describe the aligned,
inserted, and deleted note spans.

Usage example (standalone, no Django required)::

    from database.file_formats.importer.mondodi.monodi_parser import load_manuscript
    from database.file_formats.pcgts import Page, SymbolType, NoteName
    from omr.steps.tools.monodi_pattern_search.pcgts_comparator import compare_page_line

    manuscript = load_manuscript("/path/to/Pa 1213")
    chant = manuscript.chants[0]
    monodi_line = chant.page_lines[0]

    pcgts_page: Page = ...          # loaded via DatabasePage.pcgts().page
    music_line  = pcgts_page.all_music_lines()[0]

    result = compare_page_line(monodi_line, music_line)
    print(result.summary())

REST integration
----------------
Use ``PageLineComparisonView`` (registered at
``POST /api/monodi/compare_page_line/``) to run the comparison via HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from database.file_formats.importer.mondodi.monodi_parser import (
    MonodiNote,
    MonodiPageLine,
    MonodiSyllable,
)


# ---------------------------------------------------------------------------
# Note name → pitch helpers  (diatonic, C=0 … B=6)
# ---------------------------------------------------------------------------

_NOTE_DIATONIC: Dict[str, int] = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6,
}


def _monodi_pitch_key(note: MonodiNote) -> str:
    """Single-character pitch key used for SequenceMatcher alignment."""
    return note.base   # "A"–"G"


def _pcgts_pitch_key(symbol) -> Optional[str]:
    """
    Single-character pitch key for a PCGTS MusicSymbol.
    Returns None for symbols that are not pitched notes.
    """
    from database.file_formats.pcgts import SymbolType, NoteName
    if symbol.symbol_type != SymbolType.NOTE:
        return None
    if symbol.note_name == NoteName.UNDEFINED:
        return None
    # NoteName values map to 'C','D','E','F','G','A','B'
    return symbol.note_name.value.upper() if hasattr(symbol.note_name, 'value') else None


# ---------------------------------------------------------------------------
# Comparison result types
# ---------------------------------------------------------------------------

@dataclass
class NoteAlignment:
    """One aligned pair (or unmatched note) from the two sequences."""
    opcode: str                  # 'equal' | 'insert' | 'delete' | 'replace'
    monodi_notes: List[MonodiNote]    = field(default_factory=list)
    pcgts_notes:  List[Any]          = field(default_factory=list)   # MusicSymbol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opcode": self.opcode,
            "monodi_count": len(self.monodi_notes),
            "pcgts_count":  len(self.pcgts_notes),
            "monodi_pitches": [f"{n.base}{n.octave}" for n in self.monodi_notes],
            "pcgts_pitches":  [_pcgts_pitch_key(s) or "?" for s in self.pcgts_notes],
        }


@dataclass
class LineComparisonResult:
    """Result of comparing one MonodiPageLine against one PCGTS music line."""
    folio: str
    line_number: int
    monodi_text: str
    monodi_note_count: int
    pcgts_note_count:  int
    alignments: List[NoteAlignment] = field(default_factory=list)

    @property
    def equal_count(self) -> int:
        return sum(len(a.monodi_notes) for a in self.alignments if a.opcode == "equal")

    @property
    def accuracy(self) -> float:
        denom = max(self.monodi_note_count, self.pcgts_note_count)
        return self.equal_count / denom if denom > 0 else 1.0

    def summary(self) -> str:
        return (
            f"Folio {self.folio}, line {self.line_number}: "
            f"{self.monodi_note_count} monodi notes vs "
            f"{self.pcgts_note_count} PCGTS notes — "
            f"accuracy {self.accuracy:.1%} ({self.equal_count} matched)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folio":              self.folio,
            "line_number":        self.line_number,
            "monodi_text":        self.monodi_text,
            "monodi_note_count":  self.monodi_note_count,
            "pcgts_note_count":   self.pcgts_note_count,
            "equal_count":        self.equal_count,
            "accuracy":           round(self.accuracy, 4),
            "alignments":         [a.to_dict() for a in self.alignments],
        }


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def compare_page_line(
    monodi_line: MonodiPageLine,
    pcgts_music_line,          # database.file_formats.pcgts.page.Line
    *,
    include_note_types: Optional[List[str]] = None,
) -> LineComparisonResult:
    """
    Align the notes of *monodi_line* with the notes of a PCGTS *music_line*.

    Parameters
    ----------
    monodi_line:
        A :class:`~database.file_formats.importer.mondodi.monodi_parser.MonodiPageLine`
        as returned by ``MonodiChant.page_lines``.
    pcgts_music_line:
        A ``database.file_formats.pcgts.page.Line`` object (music line from
        a PCGTS page).
    include_note_types:
        Optional filter — only monodi notes whose ``note_type`` is in this
        list participate in the comparison.  Pass ``None`` to include all.

    Returns
    -------
    :class:`LineComparisonResult`
    """
    from database.file_formats.pcgts import SymbolType, NoteName

    # --- collect monodi notes ---
    monodi_notes: List[MonodiNote] = monodi_line.all_notes
    if include_note_types is not None:
        monodi_notes = [n for n in monodi_notes if n.note_type in include_note_types]

    # --- collect PCGTS notes ---
    pcgts_notes = [
        s for s in pcgts_music_line.symbols
        if s.symbol_type == SymbolType.NOTE and s.note_name != NoteName.UNDEFINED
    ]

    # --- build pitch-key strings for SequenceMatcher ---
    monodi_keys = "".join(_monodi_pitch_key(n) for n in monodi_notes)
    pcgts_keys  = "".join(_pcgts_pitch_key(s) or "?" for s in pcgts_notes)

    # --- align ---
    sm = SequenceMatcher(a=monodi_keys, b=pcgts_keys, autojunk=False, isjunk=False)
    alignments: List[NoteAlignment] = []
    for opcode, a0, a1, b0, b1 in sm.get_opcodes():
        alignments.append(NoteAlignment(
            opcode=opcode,
            monodi_notes=monodi_notes[a0:a1],
            pcgts_notes=pcgts_notes[b0:b1],
        ))

    return LineComparisonResult(
        folio=monodi_line.folio,
        line_number=monodi_line.line_number,
        monodi_text=monodi_line.text,
        monodi_note_count=len(monodi_notes),
        pcgts_note_count=len(pcgts_notes),
        alignments=alignments,
    )


# ---------------------------------------------------------------------------
# Multi-line comparison: align a full chant against a page's music lines
# ---------------------------------------------------------------------------

@dataclass
class ChantPageComparisonResult:
    """Result of matching all page_lines of a chant against an ommr4all page."""
    folio: str
    chant_id: str
    chant_textinitium: str
    line_results: List[LineComparisonResult] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        total_matched  = sum(r.equal_count        for r in self.line_results)
        total_monodi   = sum(r.monodi_note_count  for r in self.line_results)
        total_pcgts    = sum(r.pcgts_note_count   for r in self.line_results)
        denom = max(total_monodi, total_pcgts)
        return total_matched / denom if denom > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folio":             self.folio,
            "chant_id":          self.chant_id,
            "chant_textinitium": self.chant_textinitium,
            "overall_accuracy":  round(self.overall_accuracy, 4),
            "lines":             [r.to_dict() for r in self.line_results],
        }


def compare_chant_page(
    chant,                           # MonodiChant
    pcgts_page,                      # database.file_formats.pcgts.Page
    folio: str,
    *,
    include_note_types: Optional[List[str]] = None,
) -> ChantPageComparisonResult:
    """
    Compare all :class:`MonodiPageLine` objects of *chant* that belong to
    *folio* against the music lines of *pcgts_page* (in document order).

    Music lines from the PCGTS page are matched positionally (first
    page_line → first music line, second → second, …).  Lines without notes
    on either side are skipped.

    Parameters
    ----------
    chant:
        A :class:`~database.file_formats.importer.mondodi.monodi_parser.MonodiChant`.
    pcgts_page:
        A ``database.file_formats.pcgts.page.Page`` instance.
    folio:
        The folio identifier to filter page_lines by (e.g. ``"18"``).
    include_note_types:
        Optional note-type filter passed through to :func:`compare_page_line`.
    """
    from database.file_formats.pcgts import SymbolType, NoteName

    # Collect music lines that actually have notes
    pcgts_music_lines = [
        line for line in pcgts_page.all_music_lines()
        if any(
            s.symbol_type == SymbolType.NOTE and s.note_name != NoteName.UNDEFINED
            for s in line.symbols
        )
    ]

    # Collect monodi page_lines for this folio
    monodi_lines = [pl for pl in chant.page_lines if str(pl.folio) == str(folio)]

    line_results: List[LineComparisonResult] = []
    for ml, pl in zip(pcgts_music_lines, monodi_lines):
        result = compare_page_line(pl, ml, include_note_types=include_note_types)
        line_results.append(result)

    return ChantPageComparisonResult(
        folio=folio,
        chant_id=chant.meta.id,
        chant_textinitium=chant.meta.textinitium,
        line_results=line_results,
    )
