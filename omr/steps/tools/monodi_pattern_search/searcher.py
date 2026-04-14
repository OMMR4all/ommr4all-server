"""
Melodic pattern search over parsed monodi+ data.

Pattern format
--------------
Each pattern is a list of *elements*. Every element describes one note in the
match and is specified as one of:

    int                     — pitch-interval only (any connection, any note-type)
    [interval]              — same, list form
    [interval, connection]  — pitch-interval + connection filter
    [interval, connection, note_type]  — full specification

Where:
    interval    (int)  — diatonic (default) or semitone relative pitch from the
                         preceding note.  The very first element represents the
                         interval from the *anchor* note (the note just before
                         the pattern window) to the first note of the window.
    connection  (int | None)
                         None = any
                         0    = NEUME_START  (start of a spaced group)
                         1    = GAPED        (start of a nonSpaced sub-group)
                         2    = LOOPED       (ligature within a grouped cluster)
    note_type   (str | None)
                         None = any
                         "Normal" | "Liquescent" | "Oriscus" | "Strophicus" |
                         "Flat" | "Sharp" | "Natural" | "Ascending" | "Descending"

Example — search for an ascending third (diatonic) looped into a descending
second, allowing any note type:
    [[2, 2, None], [-1, None, None]]

Result format
-------------
:class:`RowMatch` holds one result per (pattern × chant × row) combination.
Results are sorted by ``count`` descending so the most common occurrences come
first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from database.file_formats.importer.mondodi.monodi_parser import (
    MonodiManuscript,
    MonodiChant,
    MonodiRow,
    MonodiNote,
    MonodiSyllable,
    CONN_NEUME_START,
    CONN_GAPED,
    CONN_LOOPED,
)


@dataclass(frozen=True)
class PatternElement:
    pitch_interval: int
    connection: Optional[int]
    note_type: Optional[str]


def _parse_element(raw: Any) -> PatternElement:
    if isinstance(raw, (int, float)):
        return PatternElement(int(raw), None, None)
    if isinstance(raw, (list, tuple)):
        interval   = int(raw[0]) if len(raw) > 0 else 0
        connection = raw[1]      if len(raw) > 1 else None
        note_type  = raw[2]      if len(raw) > 2 else None
        return PatternElement(interval, connection, note_type)
    return PatternElement(0, None, None)


def parse_pattern(raw_pattern: List[Any]) -> List[PatternElement]:
    return [_parse_element(e) for e in raw_pattern]


@dataclass
class RowMatch:
    manuscript_id: str
    manuscript_sigle: str
    chant_id: str
    chant_textinitium: str
    chant_festtag: str
    chant_feier: str
    chant_dokumenten_id: str
    folio: str
    row_number: int
    row_text: str
    count: int
    pattern: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manuscript_id":       self.manuscript_id,
            "manuscript_sigle":    self.manuscript_sigle,
            "chant_id":            self.chant_id,
            "chant_textinitium":   self.chant_textinitium,
            "chant_festtag":       self.chant_festtag,
            "chant_feier":         self.chant_feier,
            "chant_dokumenten_id": self.chant_dokumenten_id,
            "folio":               self.folio,
            "row_number":          self.row_number,
            "row_text":            self.row_text,
            "count":               self.count,
            "pattern":             self.pattern,
        }


class MonodiPatternSearcher:
    """
    Search for melodic patterns across one or more monodi+ manuscripts.

    Parameters
    ----------
    patterns:
        List of raw patterns (each is a list of pattern elements as described
        in the module docstring).
    syllable_only:
        If *True*, notes from different syllables are never matched across a
        syllable boundary.  Each syllable's notes form an independent search
        chunk.
    include_note_types:
        If given, only notes whose ``note_type`` is in this list participate in
        the search.  Pass ``None`` to include all note types.
    use_semitone_pitch:
        If *True*, intervals are computed in semitones (useful when accidentals
        such as Flat/Sharp need to be considered).  Default is diatonic
        (consistent with the existing PCGTS pattern-matching tool).
    """

    def __init__(
        self,
        patterns: List[List[Any]],
        syllable_only: bool = False,
        include_note_types: Optional[List[str]] = None,
        use_semitone_pitch: bool = False,
    ) -> None:
        self.parsed_patterns: List[List[PatternElement]] = [
            parse_pattern(p) for p in patterns
        ]
        self.raw_patterns = patterns
        self.syllable_only = syllable_only
        self.include_note_types = include_note_types
        self.use_semitone = use_semitone_pitch

    def _abs_pitch(self, note: MonodiNote) -> int:
        return note.abs_pitch_semitone if self.use_semitone else note.abs_pitch_diatonic

    def _filter_notes(self, notes: List[MonodiNote]) -> List[MonodiNote]:
        if self.include_note_types is None:
            return notes
        return [n for n in notes if n.note_type in self.include_note_types]

    def _get_chunks(self, row: MonodiRow) -> List[List[MonodiNote]]:
        """
        Return the note chunks to search within a row.

        In *syllable_only* mode every syllable's notes form a separate chunk so
        that no pattern can span a syllable boundary.  Otherwise all notes in
        the row form a single chunk.
        """
        if self.syllable_only:
            chunks = []
            for syl in row.syllables:
                filtered = self._filter_notes(syl.notes)
                if filtered:
                    chunks.append(filtered)
            return chunks
        else:
            filtered = self._filter_notes(row.all_notes)
            return [filtered] if filtered else []

    def _count_pattern_in_notes(
        self, notes: List[MonodiNote], pattern: List[PatternElement]
    ) -> int:
        """
        Count non-overlapping-start occurrences of *pattern* in *notes*.

        A pattern of length N requires at least N+1 notes (one anchor note
        before the window).  The sliding window iterates from index 1 to
        ``len(notes) - N``, inclusive.
        """
        n_pat = len(pattern)
        if n_pat == 0 or len(notes) < n_pat + 1:
            return 0

        abs_pitches = [self._abs_pitch(n) for n in notes]
        # rel_pitches[k] = interval from note k-1 to note k; rel_pitches[0] = 0
        rel_pitches = [0] + [
            abs_pitches[k] - abs_pitches[k - 1]
            for k in range(1, len(abs_pitches))
        ]

        count = 0
        for j in range(1, len(notes) - n_pat + 1):
            if self._window_matches(notes, rel_pitches, j, pattern):
                count += 1
        return count

    @staticmethod
    def _window_matches(
        notes: List[MonodiNote],
        rel_pitches: List[int],
        start: int,
        pattern: List[PatternElement],
    ) -> bool:
        for offset, p_elem in enumerate(pattern):
            idx = start + offset
            if rel_pitches[idx] != p_elem.pitch_interval:
                return False
            if p_elem.connection is not None and notes[idx].graphical_connection != p_elem.connection:
                return False
            if p_elem.note_type is not None and notes[idx].note_type != p_elem.note_type:
                return False
        return True

    def search_manuscript(self, manuscript: MonodiManuscript) -> List[RowMatch]:
        """Search all chants in *manuscript* and return matches."""
        results: List[RowMatch] = []

        for chant in manuscript.chants:
            for row in chant.rows:
                chunks = self._get_chunks(row)

                for pat_idx, pattern in enumerate(self.parsed_patterns):
                    row_count = sum(
                        self._count_pattern_in_notes(chunk, pattern)
                        for chunk in chunks
                    )
                    if row_count > 0:
                        results.append(RowMatch(
                            manuscript_id=manuscript.meta.id,
                            manuscript_sigle=manuscript.meta.quellensigle,
                            chant_id=chant.meta.id,
                            chant_textinitium=chant.meta.textinitium,
                            chant_festtag=chant.meta.festtag,
                            chant_feier=chant.meta.feier,
                            chant_dokumenten_id=chant.meta.dokumenten_id,
                            folio=chant.meta.foliostart,
                            row_number=row.row_number,
                            row_text=row.text,
                            count=row_count,
                            pattern=self.raw_patterns[pat_idx],
                        ))

        results.sort(key=lambda r: r.count, reverse=True)
        return results

    def search_manuscripts(self, manuscripts: List[MonodiManuscript]) -> List[RowMatch]:
        """Search across multiple manuscripts and return a globally sorted list."""
        all_results: List[RowMatch] = []
        for manuscript in manuscripts:
            all_results.extend(self.search_manuscript(manuscript))
        all_results.sort(key=lambda r: r.count, reverse=True)
        return all_results
