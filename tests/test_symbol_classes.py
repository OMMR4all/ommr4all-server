import unittest
from database.file_formats.pcgts.page.musicsymbol import (
    create_clef, ClefType, MusicSymbolPositionInStaff, NoteName, MusicSymbol, SymbolType,
)


class TestGClef(unittest.TestCase):
    """The G clef is the reference example of a symbol class added via the
    declarative extension procedure (see doc/adding_symbol_classes.md)."""

    def test_name_octave(self):
        # the G clef marks the G above the C that a C clef marks at the same position
        clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.LINE_2), (NoteName.G, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.SPACE_3), (NoteName.A, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.SPACE_2), (NoteName.F, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.LINE_1), (NoteName.E, 5))

        c_clef = create_clef(clef_type=ClefType.C, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertTupleEqual(c_clef.note_name_octave(MusicSymbolPositionInStaff.LINE_2), (NoteName.C, 5))

    def test_json_round_trip(self):
        clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        d = clef.to_json()
        self.assertEqual(d['type'], 'clef')
        self.assertEqual(d['clefType'], 'g')
        restored = MusicSymbol.from_json(d)
        self.assertEqual(restored.symbol_type, SymbolType.CLEF)
        self.assertEqual(restored.clef_type, ClefType.G)
        self.assertEqual(restored.position_in_staff, MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(restored.id, clef.id)


class TestUnsupportedSymbolClassTolerance(unittest.TestCase):
    """Workflow components must skip symbol classes they do not support
    instead of raising."""

    def test_pixel_classifier_label_mapping(self):
        from omr.imageoperations.music_line_operations import SymbolLabel

        g_clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(SymbolLabel.music_symbol_to_symbol_label(g_clef), SymbolLabel.BACKGROUND)

        c_clef = create_clef(clef_type=ClefType.C, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(SymbolLabel.music_symbol_to_symbol_label(c_clef), SymbolLabel.CLEF_C)
        f_clef = create_clef(clef_type=ClefType.F, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(SymbolLabel.music_symbol_to_symbol_label(f_clef), SymbolLabel.CLEF_F)

    def test_calamari_codec_encodes_new_subtype(self):
        from omr.dataset.datastructs import CalamariCodec, CalamariSequence

        codec = CalamariCodec()
        symbols = [
            create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2),
            MusicSymbol(SymbolType.NOTE, position_in_staff=MusicSymbolPositionInStaff.SPACE_3),
        ]
        seq = CalamariSequence(codec, symbols)
        # both symbols are representable: the dynamic codec learns the new clef subtype
        self.assertEqual(len(seq.calamari_str), 2)

    def test_evaluator_sequence_skips_unknown_types(self):
        from omr.steps.symboldetection.evaluator import Codec
        # symbols_to_label_sequence must not raise for any defined subtype
        codec = Codec()
        symbols = [
            create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2),
            MusicSymbol(SymbolType.NOTE, position_in_staff=MusicSymbolPositionInStaff.SPACE_3),
        ]
        seq = codec.symbols_to_label_sequence(symbols, False)
        self.assertEqual(len(seq), 2)


if __name__ == '__main__':
    unittest.main()
