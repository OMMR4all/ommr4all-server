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

from restapi.operationworker.workerresources import TRAIN_OPERATIONS, default_n_epoch, \
    InvalidTrainerParamsException, required_locks, validate_training_books
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

    def test_every_algorithm_type_resolves_a_group_and_lock(self):
        """AlgorithmPredictor.unlocked goes through group()/group_2_lock_mapping(), so an
        unmapped type would break the page selection of any workflow containing it."""
        from omr.steps.algorithmtypes import AlgorithmTypes, AlgorithmGroups
        mapped = [t for types in AlgorithmGroups.group_types_mapping().values() for t in types]
        for t in AlgorithmTypes:
            self.assertEqual(mapped.count(t), 1, "{} must be in exactly one group".format(t))
            t.group().group_2_lock_mapping()  # must not raise


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


class TestTrainingBooks(DjangoTestCase):
    """Selecting which books contribute ground truth to a training run."""

    def setUp(self):
        self.book = DatabaseBook('demo')
        self.user = User.objects.create_user('training_books_user', password='pw')
        self.admin = User.objects.create_superuser('training_books_admin', password='pw')

    # -- the locks a page must carry to be usable for a step ------------------------------

    def test_required_locks_of_the_training_operations(self):
        self.assertEqual([(l.label, l.lock) for l in required_locks(TRAIN_OPERATIONS['train_symbols'])],
                         [(Locks.SYMBOLS, True)])
        # end2end needs both, so a usable page count must be an AND, not a sum per lock
        self.assertEqual(sorted([l.label.value for l in required_locks(TRAIN_OPERATIONS['train_end2end'])]),
                         ['Symbols', 'Text'])

    # -- endpoint -------------------------------------------------------------------------

    def _books(self, operation, user):
        from rest_framework.test import APIClient
        client = APIClient()
        client.force_authenticate(user=user)
        response = client.get('/api/operation/{}/training_books'.format(operation))
        self.assertEqual(response.status_code, 200, response.content)
        return response.json()

    def test_lists_readable_books_with_usable_page_counts(self):
        body = self._books('train_symbols', self.admin)
        self.assertEqual(body['locks'], ['Symbols'])
        demo = [b for b in body['books'] if b['book'] == 'demo']
        self.assertEqual(len(demo), 1, body['books'])
        self.assertEqual(demo[0]['pages'], len(self.book.pages()))
        # only page_test_lock is locked, and it has no locked text
        self.assertEqual(demo[0]['usablePages'], 1)
        self.assertEqual(demo[0]['style'], self.book.get_meta().notationStyle)

    def test_usable_pages_are_and_ed_over_all_required_locks(self):
        body = self._books('train_end2end', self.admin)
        demo = [b for b in body['books'] if b['book'] == 'demo'][0]
        self.assertEqual(demo['usablePages'], 0)

    def test_only_books_the_user_may_read_are_listed(self):
        self.assertEqual(self._books('train_symbols', self.user)['books'], [])

    def test_unknown_operation(self):
        from rest_framework.test import APIClient
        client = APIClient()
        client.force_authenticate(user=self.admin)
        self.assertEqual(client.get('/api/operation/symbols_pc/training_books').status_code, 400)

    # -- validation of a submitted selection ------------------------------------------------

    def test_validation_accepts_readable_books(self):
        self.assertEqual(validate_training_books(self.admin, ['demo']), ['demo'])
        self.assertEqual(validate_training_books(self.user, []), [])

    def test_validation_rejects_unreadable_and_unknown_books(self):
        with self.assertRaises(InvalidTrainerParamsException):
            validate_training_books(self.user, ['demo'])
        with self.assertRaises(InvalidTrainerParamsException):
            validate_training_books(self.admin, ['does_not_exist'])

    def test_the_task_runner_validates_the_selection(self):
        body = {'trainParams': {'includeAllTrainingData': True, 'books': ['demo']}}
        runner = BookOperationView.op_to_task_runner('train_symbols', self.book, body, self.admin)
        self.assertEqual(runner.params.books, ['demo'])

        with self.assertRaises(InvalidTrainerParamsException):
            BookOperationView.op_to_task_runner('train_symbols', self.book, body, self.user)

        # status/model lookups rebuild the runner without a user and must not raise
        runner = BookOperationView.op_to_task_runner('train_symbols', self.book, body, None)
        self.assertEqual(runner.params.books, ['demo'])

    # -- resolution of the training data ----------------------------------------------------

    def _books_used(self, params, books):
        from unittest import mock
        from restapi.operationworker.taskrunners.trainerparams import TaskTrainerParams
        with mock.patch('restapi.operationworker.taskrunners.trainerparams.dataset_by_locked_pages',
                        return_value=([], [])) as m:
            TaskTrainerParams.from_dict(params).to_train_val(locks=[], books=books)
        return [b.book for b in m.call_args[0][3]]

    def test_the_trained_book_is_always_included(self):
        used = self._books_used({'includeAllTrainingData': True, 'books': ['demo']}, [DatabaseBook('other')])
        self.assertEqual(used, ['other', 'demo'])

    def test_a_selected_book_is_not_added_twice(self):
        used = self._books_used({'includeAllTrainingData': True, 'books': ['demo']}, [DatabaseBook('demo')])
        self.assertEqual(used, ['demo'])

    def test_without_the_flag_only_the_trained_book_is_used(self):
        used = self._books_used({'books': ['demo']}, [DatabaseBook('other')])
        self.assertEqual(used, ['other'])

    def test_an_empty_selection_keeps_the_legacy_all_books_behaviour(self):
        used = self._books_used({'includeAllTrainingData': True}, [DatabaseBook('other')])
        self.assertEqual(used, [b.book for b in DatabaseBook.list_available()])






class TestSkippedPagesReport(TestCase):
    """A batch prediction must survive a page that could not be predicted: the page
    is reported in `skipped_pages` instead of taking the whole task (and with it the
    rest of the workflow chain) down."""

    def _run(self, predictor_results):
        from unittest import mock
        from omr.steps.algorithmtypes import AlgorithmTypes
        from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
        from restapi.operationworker.taskrunners.taskrunnerprediction import TaskRunnerPrediction, Settings

        book = DatabaseBook('demo')
        pages = book.pages()[:2]
        runner = TaskRunnerPrediction(
            AlgorithmTypes.LAYOUT_SIMPLE_LYRICS,
            PageSelection(book, PageCount.CUSTOM, pages),
            Settings(params=AlgorithmPredictorParams(), store_to_pcgts=False),
        )

        class PredictorCls:
            @staticmethod
            def unprocessed(page): return True

            @staticmethod
            def unlocked(page): return True

        meta = mock.Mock()
        meta.predictor.return_value = PredictorCls
        predictor = mock.Mock()
        predictor.predict.return_value = iter(predictor_results)

        with mock.patch.object(runner, 'algorithm_meta', return_value=meta), \
                mock.patch('omr.steps.predictorcache.get_or_create', return_value=predictor):
            return runner.run(mock.Mock(), mock.Mock())

    def test_a_failed_page_is_reported_and_the_others_are_kept(self):
        from omr.steps.algorithm import FailedPageResult
        from unittest import mock

        ok = mock.Mock()
        ok.to_dict.return_value = {'blocks': {}}
        result = self._run([ok, FailedPageResult('page00000002', 'demo', 'ValueError: broken')])

        self.assertEqual(result['results'], [{'blocks': {}}])
        self.assertEqual(result['skipped_pages'],
                         [{'page': 'page00000002', 'book': 'demo', 'error': 'ValueError: broken'}])

    def test_without_failures_the_report_is_empty(self):
        from unittest import mock

        ok = mock.Mock()
        ok.to_dict.return_value = {'blocks': {}}
        result = self._run([ok, ok])

        self.assertEqual(len(result['results']), 2)
        self.assertEqual(result['skipped_pages'], [])
