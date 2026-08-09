import os
import unittest
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from database.file_formats.pcgts import Coords, Line
from database.file_formats.pcgts.page.staffline import StaffLine, StaffLines
from omr.steps.algorithm import FailedPageResult, PredictionCallback
from omr.steps.layout.predictor import LayoutAnalysisPredictor
from omr.steps.layout.simplelyrics.predictor import Predictor as SimpleLyricsPredictor


N_POINTS = 4  # points per staff line in the fixtures below


def staff_line(y: float, n_points: int = N_POINTS) -> StaffLine:
    return StaffLine(Coords(np.array([[x / 10, y] for x in range(n_points)], dtype=float)))


def music_line(n_staff_lines: int, first_y: float = 0.1) -> Line:
    return Line(id='ml', staff_lines=StaffLines(
        [staff_line(first_y + i * 0.01) for i in range(n_staff_lines)]))


class StaffPolygonTest(unittest.TestCase):
    """The polygon of a stave is built by concatenating four point arrays. Staves
    with fewer than three staff lines leave the inner two empty, which used to
    produce a 1-D array and crash the whole layout task (ValueError: all the
    input arrays must have same number of dimensions)."""

    pad = (0, 0.005)

    def polygon(self, n_staff_lines: int) -> Optional[Coords]:
        return SimpleLyricsPredictor._staff_polygon(music_line(n_staff_lines), self.pad)

    def test_regular_stave(self):
        for n in (3, 4, 5):
            with self.subTest(n_staff_lines=n):
                coords = self.polygon(n)
                self.assertEqual(coords.points.shape, (2 * N_POINTS + 2 * (n - 2), 2))

    def test_stave_with_less_than_three_staff_lines(self):
        for n in (1, 2):
            with self.subTest(n_staff_lines=n):
                coords = self.polygon(n)
                self.assertIsNotNone(coords)
                self.assertEqual(coords.points.shape, (2 * N_POINTS, 2))
                self.assertFalse(np.isnan(coords.points).any())

    def test_padding_is_applied_to_outer_lines(self):
        coords = self.polygon(4)
        self.assertAlmostEqual(coords.points[0][1], 0.1 - self.pad[1])
        self.assertAlmostEqual(coords.points[N_POINTS + 2][1], 0.13 + self.pad[1])

    def test_staff_lines_without_coords_are_ignored(self):
        ml = music_line(4)
        ml.staff_lines[1] = StaffLine(Coords())
        coords = SimpleLyricsPredictor._staff_polygon(ml, self.pad)
        self.assertEqual(coords.points.shape, (2 * N_POINTS + 2 * (3 - 2), 2))

        empty = Line(id='ml', staff_lines=StaffLines([StaffLine(Coords()), StaffLine(Coords())]))
        self.assertIsNone(SimpleLyricsPredictor._staff_polygon(empty, self.pad))


class RecordingCallback(PredictionCallback):
    def __init__(self):
        super().__init__()
        self.processed: List[int] = []

    def progress_updated(self, percentage: float, n_pages: int = 0, n_processed_pages: int = 0):
        self.processed.append(n_processed_pages)


def fake_pcgts(name: str):
    return SimpleNamespace(page=SimpleNamespace(
        location=SimpleNamespace(page=name, book=SimpleNamespace(book='a_book'))))


class PredictEachPageTest(unittest.TestCase):
    """A page that raises must not abort the batch: it is reported as a
    FailedPageResult so the result stream stays 1:1 with the input pages."""

    class Stub(LayoutAnalysisPredictor):
        def __init__(self, failing: List[str]):
            # deliberately not calling super(): no model is needed here
            self.failing = failing

        @staticmethod
        def meta():
            return None

        def _predict(self, pcgts_files, callback=None):
            return self._predict_each_page(pcgts_files, callback)

        def _predict_single(self, pcgts):
            if pcgts.page.location.page in self.failing:
                raise ValueError('all the input arrays must have same number of dimensions')
            return 'result for {}'.format(pcgts.page.location.page)

    def test_failing_page_does_not_abort_the_batch(self):
        pages = [fake_pcgts('page0001'), fake_pcgts('page0002'), fake_pcgts('page0003')]
        callback = RecordingCallback()
        results = list(self.Stub(['page0002'])._predict(pages, callback))

        self.assertEqual(len(results), len(pages))
        self.assertEqual(results[0], 'result for page0001')
        self.assertEqual(results[2], 'result for page0003')

        failed = results[1]
        self.assertIsInstance(failed, FailedPageResult)
        self.assertEqual(failed.page_name, 'page0002')
        self.assertEqual(failed.book_name, 'a_book')
        self.assertIn('ValueError', failed.error)
        self.assertEqual(failed.to_dict()['page'], 'page0002')

    def test_progress_reaches_all_pages_despite_failures(self):
        pages = [fake_pcgts('page0001'), fake_pcgts('page0002')]
        callback = RecordingCallback()
        list(self.Stub(['page0001', 'page0002'])._predict(pages, callback))
        self.assertEqual(callback.processed[-1], len(pages))

    def test_failed_result_never_touches_the_page(self):
        # store_to_page must be inert, otherwise a batch run would clear the
        # blocks of a page it could not predict.
        FailedPageResult('page0001', 'a_book', 'boom').store_to_page()


if __name__ == '__main__':
    unittest.main()
