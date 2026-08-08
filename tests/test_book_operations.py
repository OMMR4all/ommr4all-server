import logging
import sys
import os
from unittest import TestCase

import ommr4all.settings as settings
from database import DatabaseBook
from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from restapi.operationworker.taskrunners.pageselection import PageSelection, PageSelectionParams, PageCount

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage; the env var makes the spawned task worker
# processes (which re-import settings) use the test storage as well
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase as DjangoTestCase

from restapi.operationworker.workerresources import TRAIN_OPERATIONS, default_n_epoch
from restapi.views.bookoperations import BookOperationView


class TestBookOperations(TestCase):
    def test_page_selection(self):
        book = DatabaseBook('demo')
        p = PageSelectionParams(
            count=PageCount.ALL,
        )
        sel = PageSelection.from_params(p, book)
        self.assertListEqual([p.local_path() for p in sel.get_pages()], [p.local_path() for p in book.pages()])

    def test_pages_with_lock(self):
        book = DatabaseBook('demo')
        pages = book.pages_with_lock([LockState(Locks.STAFF_LINES, True)])
        self.assertListEqual([p.local_path() for p in pages], [book.page('page_test_lock').local_path()])

        pages = book.pages_with_lock([LockState(Locks.STAFF_LINES, False), LockState(Locks.SYMBOLS, True)])
        self.assertListEqual([p.local_path() for p in pages], [])


class TestTrainingEpochs(DjangoTestCase):
    """The epoch limit must be applied where the training request is turned into a task runner,
    not only in the endpoint that tells the client what to offer."""

    OPERATION = 'train_symbols'

    def setUp(self):
        self.book = DatabaseBook('demo')
        self.user = User.objects.create_user('train_epochs_user', password='pw')
        self.default = default_n_epoch(TRAIN_OPERATIONS[self.OPERATION])

    def _runner(self, body, user=None):
        return BookOperationView.op_to_task_runner(self.OPERATION, self.book, body,
                                                   user if user is not None else self.user)

    def test_without_a_request_the_algorithm_default_applies(self):
        runner = self._runner({'trainParams': {}})
        self.assertIsNone(runner.params.n_epoch)
        self.assertIsNone(runner.params.to_trainer_params(runner.algorithm_meta().trainer()))

    def test_lowering_is_kept(self):
        runner = self._runner({'trainParams': {'n_epoch': 5}})
        self.assertEqual(runner.params.n_epoch, 5)
        params = runner.params.to_trainer_params(runner.algorithm_meta().trainer())
        self.assertEqual(params.n_epoch, 5)
        # the other hyper parameters must stay at the algorithm defaults
        self.assertEqual(params.n_iter, runner.algorithm_meta().trainer().default_params().n_iter)

    def test_raising_is_capped_for_a_regular_user(self):
        runner = self._runner({'trainParams': {'n_epoch': self.default + 1000}})
        self.assertEqual(runner.params.n_epoch, self.default)

    def test_raising_is_allowed_for_an_admin(self):
        self.user.is_superuser = True
        self.user.save()
        runner = self._runner({'trainParams': {'n_epoch': self.default + 1000}})
        self.assertEqual(runner.params.n_epoch, self.default + 1000)




