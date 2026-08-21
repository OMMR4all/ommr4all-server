import os
import unittest
from typing import List

import numpy as np

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from database.file_formats.pcgts import Coords, Line, MusicSymbol, MusicSymbolPositionInStaff, Point, SymbolType
from database.file_formats.pcgts.page.staffline import StaffLine, StaffLines
from omr.steps.stafflines.correction.predictor import LineShift, _apply_shift, _line_samples, compute_shift


STAFF_SPACE = 20     # px between two staff lines, as in the NORMALIZED_X2 image
LINE_WIDTH = 1       # px of ink per staff line, so the true offset is unambiguous
IMAGE = (300, 400)   # h, w


def synthetic_page(first_y: int, n_lines: int = 4) -> np.ndarray:
    """A bright page with ``n_lines`` dark horizontal staff lines."""
    gray = np.full(IMAGE, 220.0)
    for i in range(n_lines):
        y = first_y + i * STAFF_SPACE
        gray[y:y + LINE_WIDTH, :] = 30.0
    return gray


def samples_for(first_y: float, n_lines: int = 4, width: int = IMAGE[1]) -> List:
    """Sampled staff lines of a stave whose lines sit at ``first_y + i * STAFF_SPACE``."""
    samples = []
    for i in range(n_lines):
        coords = Coords(np.array([[0, first_y + i * STAFF_SPACE],
                                  [width - 1, first_y + i * STAFF_SPACE]], dtype=float))
        samples.append(_line_samples(coords, width, IMAGE[0]))
    return samples


class ComputeShiftTest(unittest.TestCase):
    def test_finds_the_offset_of_a_shifted_stave(self):
        gray = synthetic_page(100)
        for offset in (-6, -3, -1, 1, 3, 6):
            with self.subTest(offset=offset):
                dy, _, _ = compute_shift(gray, samples_for(100 + offset), max_shift=10)
                self.assertEqual(-offset, dy)

    def test_keeps_a_correct_stave_where_it_is(self):
        gray = synthetic_page(100)
        dy, before, after = compute_shift(gray, samples_for(100), max_shift=10)
        self.assertEqual(0, dy)
        self.assertEqual(before, after)

    def test_does_not_move_a_stave_on_a_blank_area(self):
        # nothing to fit to: every candidate scores the same, so the stave must stay put
        gray = np.full(IMAGE, 220.0)
        dy, _, _ = compute_shift(gray, samples_for(100), max_shift=10)
        self.assertEqual(0, dy)

    def test_prefers_the_smallest_of_equally_good_shifts(self):
        # a thick band fits equally well over several shifts; the correction must be minimal
        gray = np.full(IMAGE, 220.0)
        for i in range(4):
            gray[100 + i * STAFF_SPACE - 3:100 + i * STAFF_SPACE + 4, :] = 30.0
        dy, _, _ = compute_shift(gray, samples_for(100), max_shift=10)
        self.assertEqual(0, dy)

    def test_leaves_a_stave_alone_when_the_offset_exceeds_the_window(self):
        # nothing inside the search window resembles a staff, so a half correction that
        # would end up nowhere near the ink must not be applied
        gray = synthetic_page(100)
        dy, _, _ = compute_shift(gray, samples_for(112), max_shift=3)
        self.assertEqual(0, dy)

    def test_ignores_staff_lines_outside_the_image(self):
        gray = synthetic_page(100)
        coords = Coords(np.array([[0, -50], [IMAGE[1] - 1, -50]], dtype=float))
        self.assertIsNone(_line_samples(coords, IMAGE[1], IMAGE[0]))


def music_line(first_y: float, n_lines: int = 4, n_points: int = 5) -> Line:
    staff_lines = StaffLines([
        StaffLine(Coords(np.array([[x / 10, first_y + i * 0.01] for x in range(n_points)], dtype=float)))
        for i in range(n_lines)])
    return Line(id='ml',
                coords=Coords(np.array([[0.0, first_y - 0.01], [0.4, first_y - 0.01],
                                        [0.4, first_y + 0.04], [0.0, first_y + 0.04]], dtype=float)),
                staff_lines=staff_lines,
                # the note sits on the third line from the bottom of the four line stave
                symbols=[MusicSymbol(SymbolType.NOTE, coord=Point(0.1, first_y + 0.01),
                                     position_in_staff=MusicSymbolPositionInStaff.LINE_3)])


class ApplyShiftTest(unittest.TestCase):
    def test_only_the_y_of_the_existing_points_changes(self):
        line = music_line(0.1)
        before = [sl.coords.points.copy() for sl in line.staff_lines]

        _apply_shift(line, 0.005, move_symbols=True)

        for sl, points in zip(line.staff_lines, before):
            self.assertEqual(len(points), len(sl.coords.points))
            np.testing.assert_allclose(points[:, 0], sl.coords.points[:, 0])
            np.testing.assert_allclose(points[:, 1] + 0.005, sl.coords.points[:, 1])

    def test_the_shape_of_a_warped_staff_line_survives(self):
        line = music_line(0.1)
        line.staff_lines[0].coords.points[:, 1] += np.array([0.0, 0.002, -0.001, 0.003, 0.0])
        before = line.staff_lines[0].coords.points.copy()

        _apply_shift(line, -0.004, move_symbols=True)

        np.testing.assert_allclose(np.diff(before[:, 1]),
                                   np.diff(line.staff_lines[0].coords.points[:, 1]), atol=1e-12)

    def test_moving_symbols_keeps_their_position_in_staff(self):
        line = music_line(0.1)
        symbol = line.symbols[0]
        position = symbol.position_in_staff

        _apply_shift(line, 0.005, move_symbols=True)

        self.assertAlmostEqual(0.115, symbol.coord.y)
        self.assertEqual(position, symbol.position_in_staff)
        self.assertEqual(1, len(line.symbols))

    def test_not_moving_symbols_re_derives_the_position_in_staff(self):
        line = music_line(0.1)
        symbol = line.symbols[0]

        # the stave moves up by one staff space, the symbol stays on the page and therefore
        # ends up one line lower in the staff than it was
        _apply_shift(line, -0.01, move_symbols=False)

        self.assertAlmostEqual(0.11, symbol.coord.y)
        self.assertEqual(MusicSymbolPositionInStaff.LINE_2, symbol.position_in_staff)

    def test_the_aabb_follows_the_stave(self):
        line = music_line(0.1)
        top = line.aabb.top()

        _apply_shift(line, 0.005, move_symbols=True)

        self.assertAlmostEqual(top + 0.005, line.aabb.top())


class ResultPayloadTest(unittest.TestCase):
    def test_unshifted_staves_are_not_reported(self):
        from omr.steps.stafflines.correction.predictor import StaffLineCorrectionResult

        class FakeBook:
            book = 'book'

        class FakePage:
            page = 'page'
            book = FakeBook()

        result = StaffLineCorrectionResult(
            page=FakePage(),
            shifts=[LineShift('a', 0.0, 0, 1.0, 1.0), LineShift('b', -0.002, -2, 2.0, 1.0)],
            move_symbols=True)

        self.assertEqual([{'id': 'b', 'dy': -0.002}], result.to_dict()['lines'])
        self.assertTrue(result.to_dict()['moveSymbols'])


if __name__ == '__main__':
    unittest.main()
