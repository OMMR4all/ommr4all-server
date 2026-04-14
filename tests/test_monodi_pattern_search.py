from __future__ import annotations

import os
import sys
import unittest

SERVER_ROOT = os.path.join(os.path.dirname(__file__), "..")
MOCKUP_ROOT = os.path.abspath(
    os.path.join(SERVER_ROOT, "..", "..", "mockup", "export")
)
sys.path.insert(0, SERVER_ROOT)

from database.file_formats.importer.mondodi.monodi_parser import (
    load_manuscript,
    parse_data_json,
    CONN_NEUME_START,
    CONN_GAPED,
    CONN_LOOPED,
)
from omr.steps.tools.monodi_pattern_search.searcher import (
    MonodiPatternSearcher,
    parse_pattern,
    PatternElement,
)


MANUSCRIPT_PA1213 = os.path.join(MOCKUP_ROOT, "Pa 1213")


class TestMonodiParser(unittest.TestCase):

    def _chant_data_json(self, uuid: str) -> dict:
        import json
        path = os.path.join(MANUSCRIPT_PA1213, uuid, "data.json")
        with open(path) as fh:
            return json.load(fh)

    def test_load_manuscript_finds_all_chants(self):
        if not os.path.isdir(MANUSCRIPT_PA1213):
            self.skipTest("Mockup data not present")
        manuscript = load_manuscript(MANUSCRIPT_PA1213)
        self.assertGreater(len(manuscript.chants), 0)
        print(f"\n  Loaded '{manuscript.meta.id}' — {len(manuscript.chants)} chants")

    def test_manuscript_meta(self):
        if not os.path.isdir(MANUSCRIPT_PA1213):
            self.skipTest("Mockup data not present")
        manuscript = load_manuscript(MANUSCRIPT_PA1213)
        self.assertEqual(manuscript.meta.id, "Pa 1213")
        self.assertIn("France", manuscript.meta.bibliothek)

    def test_chant_has_rows(self):
        if not os.path.isdir(MANUSCRIPT_PA1213):
            self.skipTest("Mockup data not present")
        manuscript = load_manuscript(MANUSCRIPT_PA1213)
        total_rows = sum(len(c.rows) for c in manuscript.chants)
        self.assertGreater(total_rows, 0)
        print(f"  Total rows across all chants: {total_rows}")

    def test_note_graphical_connections(self):
        data = {
            "kind": "RootContainer",
            "children": [{
                "kind": "FormteilContainer",
                "children": [{
                    "kind": "ZeileContainer",
                    "children": [{
                        "kind": "Syllable",
                        "uuid": "s1",
                        "text": "te",
                        "syllableType": "Normal",
                        "notes": {
                            "spaced": [
                                {
                                    "nonSpaced": [
                                        {"grouped": [
                                            {"uuid": "n1", "base": "G", "octave": 4,
                                             "noteType": "Normal", "liquescent": False, "focus": False},
                                            {"uuid": "n2", "base": "A", "octave": 4,
                                             "noteType": "Normal", "liquescent": False, "focus": False},
                                        ]},
                                        {"grouped": [
                                            {"uuid": "n3", "base": "F", "octave": 4,
                                             "noteType": "Normal", "liquescent": False, "focus": False},
                                        ]},
                                    ]
                                },
                                {
                                    "nonSpaced": [
                                        {"grouped": [
                                            {"uuid": "n4", "base": "E", "octave": 4,
                                             "noteType": "Normal", "liquescent": False, "focus": False},
                                        ]},
                                    ]
                                },
                            ]
                        }
                    }]
                }]
            }]
        }
        rows = parse_data_json(data)
        self.assertEqual(len(rows), 1)
        notes = rows[0].all_notes
        self.assertEqual(len(notes), 4)

        self.assertEqual(notes[0].graphical_connection, CONN_NEUME_START)  # n1
        self.assertEqual(notes[1].graphical_connection, CONN_LOOPED)       # n2 (grouped with n1)
        self.assertEqual(notes[2].graphical_connection, CONN_GAPED)        # n3 (next nonSpaced)
        self.assertEqual(notes[3].graphical_connection, CONN_NEUME_START)  # n4 (next spaced)

    def test_accidental_semitone_pitch(self):
        data = {
            "kind": "RootContainer",
            "children": [{
                "kind": "FormteilContainer",
                "children": [{
                    "kind": "ZeileContainer",
                    "children": [{
                        "kind": "Syllable",
                        "uuid": "s1", "text": "x", "syllableType": "Normal",
                        "notes": {"spaced": [{"nonSpaced": [{"grouped": [
                            {"uuid": "n1", "base": "B", "octave": 4,
                             "noteType": "Flat", "liquescent": False, "focus": False},
                            {"uuid": "n2", "base": "B", "octave": 4,
                             "noteType": "Normal", "liquescent": False, "focus": False},
                        ]}]}]}
                    }]
                }]
            }]
        }
        rows = parse_data_json(data)
        notes = rows[0].all_notes
        # Normal B4 = 4*12 + 11 = 59; Flat B4 = 58
        self.assertEqual(notes[0].abs_pitch_semitone, 58)
        self.assertEqual(notes[1].abs_pitch_semitone, 59)
        # Diatonic pitch is unaffected by accidentals
        self.assertEqual(notes[0].abs_pitch_diatonic, notes[1].abs_pitch_diatonic)


class TestPatternParsing(unittest.TestCase):

    def test_int_element(self):
        pat = parse_pattern([2])
        self.assertEqual(pat[0], PatternElement(2, None, None))

    def test_list_element_full(self):
        pat = parse_pattern([[2, 2, "Liquescent"]])
        self.assertEqual(pat[0], PatternElement(2, 2, "Liquescent"))

    def test_list_element_partial(self):
        pat = parse_pattern([[2, 1]])
        self.assertEqual(pat[0], PatternElement(2, 1, None))

    def test_none_connection(self):
        pat = parse_pattern([[2, None, None]])
        self.assertIsNone(pat[0].connection)

def _make_notes(specs):
    """
    Build a list of MonodiNote objects from compact specs.

    Each spec is (base, octave, note_type, connection) or (base, octave).
    """
    from database.file_formats.importer.mondodi.monodi_parser import MonodiNote
    notes = []
    for i, spec in enumerate(specs):
        base, octave = spec[0], spec[1]
        note_type = spec[2] if len(spec) > 2 else "Normal"
        conn      = spec[3] if len(spec) > 3 else CONN_NEUME_START
        notes.append(MonodiNote(
            uuid=str(i), base=base, octave=octave,
            note_type=note_type, liquescent=False,
            graphical_connection=conn,
        ))
    return notes


class TestSearcher(unittest.TestCase):

    def _searcher(self, patterns, **kwargs):
        return MonodiPatternSearcher(patterns=patterns, **kwargs)

    def test_simple_interval_match(self):
        notes = _make_notes([("C", 4), ("E", 4), ("G", 4)])
        searcher = self._searcher([[2, 2]])
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 1)

    def test_no_match(self):
        notes = _make_notes([("C", 4), ("D", 4), ("E", 4)])
        searcher = self._searcher([[3]])  # interval 3 not present
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 0)

    def test_connection_filter_match(self):
        notes = _make_notes([
            ("C", 4, "Normal", CONN_NEUME_START),
            ("E", 4, "Normal", CONN_LOOPED),
            ("G", 4, "Normal", CONN_NEUME_START),
        ])
        searcher = self._searcher([[[2, CONN_LOOPED, None], [2, None, None]]])
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 1)

    def test_connection_filter_no_match(self):
        notes = _make_notes([
            ("C", 4, "Normal", CONN_NEUME_START),
            ("E", 4, "Normal", CONN_GAPED),
            ("G", 4, "Normal", CONN_NEUME_START),
        ])
        searcher = self._searcher([[[2, CONN_LOOPED]]])
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 0)

    def test_note_type_filter(self):
        notes = _make_notes([
            ("C", 4, "Normal",    CONN_NEUME_START),
            ("E", 4, "Liquescent", CONN_NEUME_START),
            ("G", 4, "Normal",    CONN_NEUME_START),
        ])
        searcher = self._searcher([[[2, None, "Liquescent"], [2, None, None]]])
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 1)

    def test_note_type_filter_no_match(self):
        notes = _make_notes([
            ("C", 4, "Normal", CONN_NEUME_START),
            ("E", 4, "Normal", CONN_NEUME_START),
        ])
        searcher = self._searcher([[[2, None, "Liquescent"]]])
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 0)

    def test_semitone_accidental(self):
        notes = _make_notes([
            ("C", 4),               # semitone abs: 48
            ("B", 4, "Flat"),       # semitone abs: 58 → interval = +10
            ("B", 4, "Normal"),     # semitone abs: 59 → interval = +1
        ])
        searcher = self._searcher([[10, 1]], use_semitone_pitch=True)
        count = searcher._count_pattern_in_notes(notes, searcher.parsed_patterns[0])
        self.assertEqual(count, 1)

    def test_include_note_types_filter(self):
        notes = _make_notes([
            ("C", 4, "Normal"),
            ("E", 4, "Flat"),     # excluded
            ("G", 4, "Normal"),
        ])
        searcher_all = self._searcher([[2, 2]])
        self.assertEqual(
            searcher_all._count_pattern_in_notes(notes, searcher_all.parsed_patterns[0]),
            1,
        )
        searcher_normal = self._searcher([[2, 2]], include_note_types=["Normal"])
        filtered = searcher_normal._filter_notes(notes)
        self.assertEqual(len(filtered), 2)  # only C and G remain
        count = searcher_normal._count_pattern_in_notes(
            filtered, searcher_normal.parsed_patterns[0]
        )
        self.assertEqual(count, 0)

    def test_syllable_only_blocks_cross_boundary(self):
        from database.file_formats.importer.mondodi.monodi_parser import (
            MonodiRow, MonodiSyllable,
        )
        syl1_notes = _make_notes([("C", 4), ("E", 4)])
        syl2_notes = _make_notes([("G", 4), ("A", 4)])
        row = MonodiRow(
            row_number=1,
            syllables=[
                MonodiSyllable("s1", "te", "Normal", syl1_notes),
                MonodiSyllable("s2", "ste", "Normal", syl2_notes),
            ],
        )
        searcher_full = self._searcher([[2, 2]], syllable_only=False)
        chunks_full = searcher_full._get_chunks(row)
        count_full = sum(
            searcher_full._count_pattern_in_notes(ch, searcher_full.parsed_patterns[0])
            for ch in chunks_full
        )
        self.assertEqual(count_full, 1)

        searcher_syl = self._searcher([[2, 2]], syllable_only=True)
        chunks_syl = searcher_syl._get_chunks(row)
        count_syl = sum(
            searcher_syl._count_pattern_in_notes(ch, searcher_syl.parsed_patterns[0])
            for ch in chunks_syl
        )
        self.assertEqual(count_syl, 0)

    def test_search_manuscript_integration(self):
        if not os.path.isdir(MANUSCRIPT_PA1213):
            self.skipTest("Mockup data not present")
        manuscript = load_manuscript(MANUSCRIPT_PA1213)

        searcher = MonodiPatternSearcher(patterns=[[0]])
        results = searcher.search_manuscript(manuscript)

        print(f"\n  Unison pattern: {len(results)} matching rows")
        for r in results[:5]:
            print(f"    [{r.chant_textinitium}] row {r.row_number} — "
                  f"folio {r.folio} — count {r.count}")
            print(f"      text: {r.row_text[:60]}")

        self.assertGreater(len(results), 0)
        counts = [r.count for r in results]
        self.assertEqual(counts, sorted(counts, reverse=True))



if __name__ == "__main__":
    unittest.main(verbosity=2)
