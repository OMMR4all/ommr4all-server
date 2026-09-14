import os
import unittest

from omr.dataset import DatasetParams
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.imageoperations.symbol_heads import AdditionalSymbolLabel
from omr.imageoperations.symbol_label_set import SymbolClassLabelSets, color_for_custom_label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBuiltinLabelSets(unittest.TestCase):
    """The built-in label sets must stay byte-identical to the two frozen enums, because
    every already-trained checkpoint was built against them."""

    def test_main_color_map_matches_the_symbol_label_enum(self):
        color_map = SymbolClassLabelSets.builtin().main_color_map()
        self.assertEqual(
            [(c.label, c.name, c.color) for c in color_map.class_spec],
            [(i.value, i.name.lower(), i.get_color()) for i in SymbolLabel])

    def test_note_type_color_map_matches_the_additional_label_enum(self):
        color_map = SymbolClassLabelSets.builtin().note_type_color_map()
        self.assertEqual(
            [(c.label, c.name, c.color) for c in color_map.class_spec],
            [(i.value, i.name.lower(), i.get_color()) for i in AdditionalSymbolLabel])

    def test_custom_colors_never_collide_with_a_builtin_one(self):
        builtin = [l.get_color() for l in SymbolLabel] + [l.get_color() for l in AdditionalSymbolLabel]
        colors = [color_for_custom_label(n) for n in range(32)]
        for c in colors:
            self.assertNotIn(c, builtin)
        self.assertEqual(len(colors), len(set(map(tuple, colors))))


class TestDatasetParamsBackwardCompatibility(unittest.TestCase):
    """A model trained before symbol classes existed records no label set and must keep
    loading with its calamari codec intact."""

    def test_shipped_sequence_to_sequence_params_still_parse(self):
        path = os.path.join(BASE_DIR, 'internal_storage', 'default_models', 'french14',
                            'symbols_sequence_to_sequence_guppy', 'dataset_params.json')
        with open(path) as f:
            params = DatasetParams.from_json(f.read())
        self.assertIsNone(params.symbol_label_sets)
        self.assertEqual(params.calamari_codec.last_char, 127)
        self.assertEqual(len(params.calamari_codec.decodec), 67)
        # the extra CodecType field must not change an existing tuple
        self.assertTrue(all(c.symbol_class is None for c in params.calamari_codec.decodec.values()))


if __name__ == '__main__':
    unittest.main()
