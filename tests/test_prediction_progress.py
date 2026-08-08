import os
import sys
import unittest
from typing import List, Tuple

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from omr.steps.algorithm import PredictionProgress
from restapi.operationworker.task import TaskStatus, TaskStatusCodes, TaskProgressCodes
from restapi.operationworker.taskqueue import TaskQueue


class RecordingCallback:
    """Captures everything a predictor reports, in order."""

    def __init__(self):
        self.log: List[Tuple[float, int, int]] = []

    def progress_updated(self, percentage: float, n_pages: int = 0, n_processed_pages: int = 0):
        self.log.append((percentage, n_processed_pages, n_pages))

    @property
    def percentages(self) -> List[float]:
        return [p for p, _, _ in self.log]

    @property
    def processed(self) -> List[int]:
        return [n for _, n, _ in self.log]


class PredictionProgressTest(unittest.TestCase):
    def assertMonotonic(self, callback: RecordingCallback):
        self.assertEqual(callback.percentages, sorted(callback.percentages),
                         'percentage moved backwards: {}'.format(callback.percentages))
        self.assertEqual(callback.processed, sorted(callback.processed),
                         'n_processed_pages moved backwards: {}'.format(callback.processed))

    def test_staff_line_pattern_is_monotonic(self):
        """The exact shape that used to make the bar oscillate.

        Staff line detection interleaves sub-page steps (from the line-detection
        library's own callback) with page completions. Before the fix these were
        two independent emitters on different scales, so each page boundary
        dropped the bar and reset the label to 0/N.
        """
        callback = RecordingCallback()
        progress = PredictionProgress(callback, 5)
        progress.start()
        for _ in range(5):
            for step in range(1, 6):
                progress.sub_progress(step / 5)
            progress.page_finished()

        self.assertMonotonic(callback)
        self.assertEqual(callback.log[0], (0.0, 0, 5))
        self.assertEqual(callback.log[-1], (1.0, 5, 5))
        # The label must never fall back to 0 once a page has completed.
        self.assertNotIn(0, callback.processed[7:])

    def test_sub_progress_never_exceeds_the_current_page(self):
        callback = RecordingCallback()
        progress = PredictionProgress(callback, 4)
        progress.page_finished()
        progress.sub_progress(1.0)
        # One page done plus a full page of sub-progress is 2/4, not more.
        self.assertEqual(callback.log[-1][0], 0.5)
        self.assertEqual(callback.log[-1][1], 1)

    def test_only_page_finished_advances_the_page_counter(self):
        callback = RecordingCallback()
        progress = PredictionProgress(callback, 3)
        for step in range(1, 4):
            progress.sub_progress(step / 3)
        self.assertEqual(set(callback.processed), {0})
        progress.page_finished()
        self.assertEqual(callback.processed[-1], 1)

    def test_out_of_order_sub_progress_cannot_drag_the_bar_back(self):
        callback = RecordingCallback()
        progress = PredictionProgress(callback, 2)
        progress.sub_progress(0.8)
        progress.sub_progress(0.1)
        self.assertMonotonic(callback)

    def test_line_based_reports_page_counts_not_line_counts(self):
        """Line-granular percentage, page-granular label."""
        callback = RecordingCallback()
        # 3 pages holding 4, 1 and 3 lines respectively
        progress = PredictionProgress(callback, 3, item_pages=[0, 0, 0, 0, 1, 2, 2, 2])
        progress.start()
        for i in range(8):
            progress.item_finished(i)

        self.assertMonotonic(callback)
        self.assertEqual(callback.log[-1], (1.0, 3, 3))
        # n_pages is the page count, never the 8 lines.
        self.assertTrue(all(n_pages == 3 for _, _, n_pages in callback.log))

    def test_pages_without_items_still_reach_completion(self):
        callback = RecordingCallback()
        # page 1 contributes no lines at all
        progress = PredictionProgress(callback, 3, item_pages=[0, 0, 2, 2])
        progress.start()
        for i in range(4):
            progress.item_finished(i)
        self.assertEqual(callback.log[-1], (1.0, 3, 3))

    def test_no_callback_is_a_silent_noop(self):
        progress = PredictionProgress(None, 5)
        progress.start()
        progress.sub_progress(0.5)
        progress.page_finished()
        progress.item_finished(0)


class TaskQueueMonotonicGuardTest(unittest.TestCase):
    """The server-side backstop in TaskQueue.update_status."""

    @staticmethod
    def running(progress: float, n_processed: int,
                progress_code=TaskProgressCodes.WORKING) -> TaskStatus:
        return TaskStatus(TaskStatusCodes.RUNNING, progress_code,
                          progress=progress, n_processed=n_processed, n_total=10)

    def test_backwards_progress_is_held_at_the_high_water_mark(self):
        merged = TaskQueue._monotonic(self.running(0.5, 5), self.running(0.13, 0))
        self.assertEqual(merged.progress, 0.5)
        self.assertEqual(merged.n_processed, 5)

    def test_forward_progress_passes_through(self):
        merged = TaskQueue._monotonic(self.running(0.5, 5), self.running(0.6, 6))
        self.assertEqual(merged.progress, 0.6)
        self.assertEqual(merged.n_processed, 6)

    def test_progress_code_change_restarts_the_scale(self):
        previous = self.running(0.9, 9, TaskProgressCodes.LOADING_DATA)
        merged = TaskQueue._monotonic(previous, self.running(0.0, 0, TaskProgressCodes.WORKING))
        self.assertEqual(merged.progress, 0.0)
        self.assertEqual(merged.n_processed, 0)

    def test_terminal_status_is_published_verbatim(self):
        previous = self.running(0.7, 7)
        finished = TaskStatus(TaskStatusCodes.FINISHED)
        self.assertIs(TaskQueue._monotonic(previous, finished), finished)


if __name__ == '__main__':
    unittest.main()
