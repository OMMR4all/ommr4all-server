import os
import unittest

import numpy as np

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from database.database_book_meta import DatabaseBookMeta
from database.file_formats.pcgts import Coords, Point
from database.file_formats.pcgts.page.pitchparams import PitchDetectionParams
from database.file_formats.pcgts.page.staffline import StaffLine, StaffLines


STAFF_SPACE = 0.1    # distance between two adjacent staff lines
TOP_Y = 0.1          # y of the topmost staff line of the synthetic stave


def staff(n_lines: int = 4) -> StaffLines:
    return StaffLines([StaffLine(Coords(np.array([[0.0, TOP_Y + i * STAFF_SPACE],
                                                  [1.0, TOP_Y + i * STAFF_SPACE]])))
                       for i in range(n_lines)])


def former_round_to_staff_pos(x: float):
    """The tolerance rule as it was hardcoded before it became configurable."""
    rounded = np.round(x)
    even = (rounded + 2000) % 2 == 0
    if not even:
        if abs(x - rounded) < 0.4:
            return rounded
        return rounded + 1 if x - rounded > 0 else rounded - 1
    return rounded


class PitchDetectionParamsTest(unittest.TestCase):
    def test_defaults_reproduce_the_former_hardcoded_tolerance(self):
        sl = staff()
        for i in range(-4000, 4000):
            x = i / 100.0
            frac = (x / 2) % 1
            if abs(frac - 0.3) < 1e-9 or abs(frac - 0.7) < 1e-9:
                continue    # exactly on a boundary, where the tie is resolved the other way
            self.assertEqual(former_round_to_staff_pos(x), sl._round_to_staff_pos(x),
                             'differs at x={}'.format(x))

    def test_default_bands_are_symmetric(self):
        sl = staff()
        # y measured downwards, so 0.25 of the gap below the second line is 0.75 above the third
        gap_top = TOP_Y + STAFF_SPACE
        for frac, on_line in [(0.0, True), (0.25, True), (0.5, False), (0.75, True), (1.0, True)]:
            pos = sl.position_in_staff(Point(0.5, gap_top + frac * STAFF_SPACE))
            self.assertEqual(on_line, pos % 2 == 1, 'frac={}'.format(frac))

    def test_asymmetric_tolerances_move_the_boundaries(self):
        sl = staff()
        gap_top = TOP_Y + STAFF_SPACE
        # the upper line of the gap claims 45%, the lower one 15% -> the space is [0.45, 0.85]
        sl.pitch_params = PitchDetectionParams(toleranceTop=0.45, toleranceBottom=0.15)
        for frac, on_line in [(0.2, True), (0.4, True), (0.5, False), (0.8, False), (0.9, True)]:
            pos = sl.position_in_staff(Point(0.5, gap_top + frac * STAFF_SPACE))
            self.assertEqual(on_line, pos % 2 == 1, 'frac={}'.format(frac))

    def test_a_position_in_a_space_becomes_a_line_position(self):
        sl = staff()
        gap_top = TOP_Y + STAFF_SPACE
        p = Point(0.5, gap_top + 0.35 * STAFF_SPACE)
        self.assertEqual(0, sl.position_in_staff(p) % 2, 'with the defaults this is a space')
        sl.pitch_params = PitchDetectionParams(toleranceTop=0.45, toleranceBottom=0.15)
        self.assertEqual(1, sl.position_in_staff(p) % 2, 'the upper line now claims it')

    def test_clef_is_forced_onto_a_line_only_while_enabled(self):
        sl = staff()
        p = Point(0.5, TOP_Y + 1.5 * STAFF_SPACE)      # centre of a space
        self.assertEqual(1, sl.position_in_staff(p, clef=True) % 2)
        sl.pitch_params = PitchDetectionParams(forceClefsOnLine=False)
        self.assertEqual(0, sl.position_in_staff(p, clef=True) % 2)

    def test_invalid_values_are_clamped_so_a_space_survives(self):
        clamped = PitchDetectionParams(0.8, 0.8).clamped()
        self.assertAlmostEqual(0.45, clamped.toleranceTop)
        self.assertAlmostEqual(0.45, clamped.toleranceBottom)

        clamped = PitchDetectionParams(-1, 5).clamped()
        self.assertAlmostEqual(0.0, clamped.toleranceTop)
        self.assertAlmostEqual(0.9, clamped.toleranceBottom)

        sl = staff()
        sl.pitch_params = PitchDetectionParams(0.8, 0.8)
        self.assertAlmostEqual(0.45, sl.pitch_params.toleranceTop, msg='clamped on assignment')

    def test_sorted_keeps_the_parameters(self):
        sl = staff()
        sl.pitch_params = PitchDetectionParams(0.45, 0.15)
        self.assertAlmostEqual(0.45, sl.sorted().pitch_params.toleranceTop)

    def test_book_meta_defaults_and_roundtrip(self):
        meta = DatabaseBookMeta.from_json('{"id": "x", "name": "x"}')
        self.assertEqual(PitchDetectionParams(), meta.pitchDetectionParams,
                         'a book_meta.json written before this field keeps the old behaviour')

        meta.pitchDetectionParams = PitchDetectionParams(0.45, 0.15, False)
        self.assertEqual(meta.pitchDetectionParams,
                         DatabaseBookMeta.from_json(meta.to_json()).pitchDetectionParams)


if __name__ == '__main__':
    unittest.main()
