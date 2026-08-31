import json
import os
import queue
import shutil
import unittest

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from database import DatabaseBook
from database.database_book_meta import DatabaseBookMeta
from database.file_formats.pcgts.page.pitchparams import PitchDetectionParams
from database.file_formats.performance.pageprogress import Locks
from restapi.operationworker.task import Task, TaskStatus, TaskStatusCodes
from restapi.operationworker.taskrunners.taskrunnerpositioninstaff import TaskRunnerPositionInStaff


TEST_BOOK = 'pis_task_test'
SOURCE_BOOK = 'demo'


def run_task(book: DatabaseBook) -> dict:
    runner = TaskRunnerPositionInStaff(book)
    task = Task('test', runner, TaskStatus(TaskStatusCodes.RUNNING), {}, None)
    return runner.run(task, queue.Queue())


def positions_of(book: DatabaseBook):
    positions = {}
    for page in book.pages():
        positions[page.page] = [int(s.position_in_staff)
                                for line in page.pcgts().page.all_music_lines() for s in line.symbols]
    return positions


class PositionInStaffTaskTest(unittest.TestCase):
    """Runs the task on a throwaway copy of the pcgts files of the demo book."""

    @classmethod
    def setUpClass(cls):
        source = DatabaseBook(SOURCE_BOOK)
        cls.book = DatabaseBook(TEST_BOOK, skip_validation=True)
        shutil.rmtree(cls.book.local_path(), ignore_errors=True)
        # the pcgts alone is not enough: parsing one reads the size of the original image
        shutil.copytree(source.local_path(), cls.book.local_path())

        cls.pages_with_symbols = [page.page for page in cls.book.pages()
                                  if any(line.symbols for line in page.pcgts().page.all_music_lines())]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.book.local_path(), ignore_errors=True)

    def set_params(self, params: PitchDetectionParams):
        meta = DatabaseBookMeta.load(self.book)
        meta.pitchDetectionParams = params
        meta.to_file(self.book)

    def tearDown(self):
        self.set_params(PitchDetectionParams())

    def test_the_fixture_has_symbols_to_work_with(self):
        self.assertGreater(len(self.pages_with_symbols), 0)

    def test_the_operation_is_wired_up(self):
        from restapi.views.bookoperations import BookOperationView
        runner = BookOperationView.op_to_task_runner('reapply_position_in_staff', self.book, {})
        self.assertIsInstance(runner, TaskRunnerPositionInStaff)
        self.assertEqual((self.book.book, ), runner.identifier())

    def test_unchanged_parameters_rewrite_nothing(self):
        paths = [p.file('pcgts').local_path() for p in self.book.pages()]
        before = {path: os.stat(path).st_mtime_ns for path in paths}

        result = run_task(self.book)

        self.assertEqual(0, result['n_updated'], 'the stored positions already match the parameters')
        self.assertEqual(0, result['n_symbols_changed'])
        self.assertEqual(0, result['n_failed'])
        self.assertGreater(result['n_total'], 0)
        for path, mtime in before.items():
            self.assertEqual(mtime, os.stat(path).st_mtime_ns,
                             'a page without changes must not be written: {}'.format(path))

    def test_changed_parameters_are_applied_and_can_be_reverted(self):
        before = positions_of(self.book)

        # the upper staff line of every gap swallows almost the whole space
        self.set_params(PitchDetectionParams(toleranceTop=0.85, toleranceBottom=0.05))
        result = run_task(self.book)
        self.assertGreater(result['n_updated'], 0)
        self.assertGreater(result['n_symbols_changed'], 0)
        self.assertEqual(0, result['n_failed'])
        self.assertNotEqual(before, positions_of(self.book))

        # the positions are a pure function of the coordinates and the parameters
        self.set_params(PitchDetectionParams())
        run_task(self.book)
        self.assertEqual(before, positions_of(self.book))

    def test_locked_pages_are_skipped(self):
        page = self.book.page(self.pages_with_symbols[0])
        progress = page.page_progress()
        progress.locked[Locks.SYMBOLS] = True
        page.save_page_progress()
        before = json.load(open(page.file('pcgts').local_path()))

        self.set_params(PitchDetectionParams(toleranceTop=0.85, toleranceBottom=0.05))
        result = run_task(self.book)

        self.assertGreaterEqual(result['n_skipped'], 1)
        self.assertEqual(before, json.load(open(page.file('pcgts').local_path())),
                         'a locked page keeps the positions its user confirmed')

        self.set_params(PitchDetectionParams())
        run_task(self.book)


if __name__ == '__main__':
    unittest.main()
