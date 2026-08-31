import json
import logging
import shutil
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
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from database.database_permissions import BookPermissionFlags, DatabaseBookPermissionFlag
from restapi.models.error import ErrorCodes
from restapi.operationworker.workerresources import TRAIN_OPERATIONS, default_n_epoch, \
    InvalidTrainerParamsException, required_locks, validate_training_books
from restapi.views.bookoperations import BookOperationView, book_operation_locked


class TestBookOperations(TestCase):
    def test_page_selection(self):
        book = DatabaseBook('demo')
        p = PageSelectionParams(
            count=PageCount.ALL,
        )
        sel = PageSelection.from_params(p, book)
        self.assertListEqual([p.local_path() for p in sel.get_pages()], [p.local_path() for p in book.pages()])

    def test_single_page_selection_keeps_the_posted_pcgts(self):
        """The editor posts its unsaved pcgts with every single-page operation
        (OperationView.op_to_task_runner). get_pages() must hand that very object to the
        predictor -- replacing it with an index-built page made every predictor re-read
        pcgts.json, so chaining layout -> symbols ran on the last saved state."""
        book = DatabaseBook('demo')
        page = book.pages()[0]
        posted = page.pcgts_from_dict(page.pcgts().to_json())

        pages = PageSelection.from_page(page).get_pages()
        self.assertEqual(len(pages), 1)
        self.assertIs(pages[0].pcgts(), posted)
        # the progress is still filled in from the index (that is what the pass is for)
        self.assertIsNotNone(pages[0].page_progress())

    def test_single_page_selection_keeps_verified_pages(self):
        """A verified page re-run from the editor is deliberate; filtering it out here
        surfaces as 'produced no result for page ...'."""
        book = DatabaseBook('demo')
        page = book.pages()[0]
        progress = page.page_progress()
        was_verified = progress.verified
        progress.verified = True
        page.set_page_progress(progress)
        try:
            self.assertEqual([p.page for p in PageSelection.from_page(page).get_pages()], [page.page])
        finally:
            progress.verified = was_verified
            page.set_page_progress(progress)

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


class TestBookOperationLocks(APITestCase):
    """A maintainer may reserve the book wide runs resp. the trainings of a book for
    maintainers (DatabaseBookMeta.lockBookOperations / lockTraining). Write access then no
    longer starts them, while editing pages and the single page algorithms of the editor --
    which never touch these endpoints -- stay available."""

    BOOK = 'operation_lock_test'
    OPERATION = 'staffs'          # any prediction; only the name is used here
    TRAIN_OPERATION = 'train_symbols'

    def setUp(self):
        # a scratch book: a lock left behind in the shared 'demo' fixture would break
        # every other test that starts an operation on it
        self.root = os.path.join(settings.PRIVATE_MEDIA_ROOT, self.BOOK)
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(os.path.join(self.root, 'pages'))
        self.book = DatabaseBook(self.BOOK)

        self.writer = User.objects.create_user(username='lock_writer', password='pw')
        self.maintainer = User.objects.create_user(username='lock_maintainer', password='pw')
        permissions = self.book.get_permissions()
        permissions.get_or_add_user_permissions(
            'lock_writer', BookPermissionFlags(DatabaseBookPermissionFlag.READ_WRITE))
        permissions.get_or_add_user_permissions(
            'lock_maintainer', BookPermissionFlags(DatabaseBookPermissionFlag.READ_WRITE
                                                   | DatabaseBookPermissionFlag.EDIT_BOOK_META))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # helpers ------------------------------------------------------------------

    def _login(self, username):
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': username, 'password': 'pw'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer {0}'.format(response.data['access']))

    def _set_locks(self, book_operations=None, training=None):
        meta = self.book.get_meta()
        meta.lockBookOperations = book_operations
        meta.lockTraining = training
        meta.to_file(self.book)

    # the meta field ------------------------------------------------------------

    def test_a_book_without_the_fields_is_unlocked(self):
        meta = self.book.get_meta()
        self.assertIsNone(meta.lockBookOperations)
        self.assertIsNone(meta.lockTraining)
        self.assertFalse(meta.book_operations_locked)
        self.assertFalse(meta.training_locked)
        self.assertFalse(book_operation_locked(self.book, self.writer, self.OPERATION))

    def test_a_meta_put_without_the_fields_keeps_the_locks(self):
        """A client that does not know the locks must not unlock the book by saving the
        book settings -- absent is not False."""
        self._set_locks(book_operations=True, training=True)
        self._login('lock_maintainer')

        meta = self.book.get_meta().to_dict()
        del meta['lockBookOperations']
        del meta['lockTraining']
        response = self.client.put('/api/book/{}/meta'.format(self.BOOK), meta, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

        self.assertTrue(DatabaseBook(self.BOOK).get_meta().book_operations_locked)
        self.assertTrue(DatabaseBook(self.BOOK).get_meta().training_locked)

    def test_a_meta_put_can_unlock(self):
        self._set_locks(book_operations=True, training=True)
        self._login('lock_maintainer')

        meta = self.book.get_meta().to_dict()
        meta['lockBookOperations'] = False
        meta['lockTraining'] = False
        response = self.client.put('/api/book/{}/meta'.format(self.BOOK), meta, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

        self.assertFalse(DatabaseBook(self.BOOK).get_meta().book_operations_locked)
        self.assertFalse(DatabaseBook(self.BOOK).get_meta().training_locked)

    # who is barred -------------------------------------------------------------

    def test_the_two_locks_are_independent(self):
        self._set_locks(book_operations=True, training=False)
        self.assertTrue(book_operation_locked(self.book, self.writer, self.OPERATION))
        self.assertFalse(book_operation_locked(self.book, self.writer, self.TRAIN_OPERATION))

        self._set_locks(book_operations=False, training=True)
        self.assertFalse(book_operation_locked(self.book, self.writer, self.OPERATION))
        self.assertTrue(book_operation_locked(self.book, self.writer, self.TRAIN_OPERATION))

    def test_a_maintainer_is_never_locked_out(self):
        self._set_locks(book_operations=True, training=True)
        for operation in (self.OPERATION, self.TRAIN_OPERATION):
            self.assertFalse(book_operation_locked(self.book, self.maintainer, operation))

    def test_reapplying_the_position_in_staff_follows_the_book_lock(self):
        """It rewrites the pitch of every symbol of the book, so it belongs to the book
        wide runs even though the settings tab that offers it is maintainer only -- the
        endpoint itself only asks for READ_WRITE."""
        self._set_locks(book_operations=True)
        self.assertTrue(book_operation_locked(self.book, self.writer, 'reapply_position_in_staff'))
        self.assertFalse(book_operation_locked(self.book, self.maintainer, 'reapply_position_in_staff'))

        self._login('lock_writer')
        response = self.client.put(
            '/api/book/{}/operation/reapply_position_in_staff/'.format(self.BOOK), {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_OPERATIONS_LOCKED.value)

    def test_an_export_is_never_locked(self):
        """Exports do not modify the book, and locking them would take away a writer's
        only way to get the book out."""
        self._set_locks(book_operations=True, training=True)
        self.assertFalse(book_operation_locked(self.book, self.writer, 'documents_export'))

    # the endpoint --------------------------------------------------------------

    def test_a_writer_may_not_start_a_locked_run(self):
        self._set_locks(book_operations=True)
        self._login('lock_writer')

        response = self.client.put('/api/book/{}/operation/{}/'.format(self.BOOK, self.OPERATION),
                                   {'selection': {'count': 'all'}}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_OPERATIONS_LOCKED.value)

    def test_a_writer_may_not_start_a_locked_training(self):
        self._set_locks(training=True)
        self._login('lock_writer')

        response = self.client.put(
            '/api/book/{}/operation/{}/'.format(self.BOOK, self.TRAIN_OPERATION),
            {'trainParams': {}}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_OPERATIONS_LOCKED.value)

    def test_a_writer_may_not_cancel_a_locked_run(self):
        """The point of the lock is that a maintainer's run survives -- being unable to
        start one but able to stop it would be pointless."""
        self._set_locks(book_operations=True)
        self._login('lock_writer')

        response = self.client.delete(
            '/api/book/{}/operation/{}/task/some-task-id'.format(self.BOOK, self.OPERATION))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_OPERATIONS_LOCKED.value)

    def test_a_writer_may_not_delete_a_model_of_a_training_locked_book(self):
        self._set_locks(training=True)
        self._login('lock_writer')

        response = self.client.delete(
            '/api/book/{}/operation/symbols_pc_torch/model/any'.format(self.BOOK))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_OPERATIONS_LOCKED.value)

    def test_a_reader_is_still_rejected_for_lacking_rights(self):
        """The lock is an extra hurdle, not a replacement for the permission check."""
        User.objects.create_user(username='lock_reader', password='pw')
        self.book.get_permissions().get_or_add_user_permissions(
            'lock_reader', BookPermissionFlags(DatabaseBookPermissionFlag.READ))
        self._login('lock_reader')

        response = self.client.put('/api/book/{}/operation/{}/'.format(self.BOOK, self.OPERATION),
                                   {'selection': {'count': 'all'}}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_INSUFFICIENT_RIGHTS.value)


class TestTaskListsWithoutAnAlgorithm(DjangoTestCase):
    """Not every task is an algorithm: the export and the position in staff runner have no
    AlgorithmTypes. Reporting `algorithm_type.value` for them made every task list endpoint
    answer 500 as long as such a task was queued, which also hid all the other tasks."""

    def test_the_runners_report_their_rest_operation_name(self):
        from restapi.operationworker.taskrunners.taskrunnerpositioninstaff import TaskRunnerPositionInStaff
        from restapi.operationworker.taskrunners.taskrunnerdocumentsexport import TaskRunnerDocumentsExport

        book = DatabaseBook('demo')
        self.assertIsNone(TaskRunnerPositionInStaff(book).algorithm_type)
        self.assertEqual(TaskRunnerPositionInStaff(book).operation(), 'reapply_position_in_staff')
        self.assertEqual(TaskRunnerDocumentsExport(book, TaskRunnerDocumentsExport.FORMAT_MEI4_ZIP).operation(),
                         'documents_export')

    def test_an_algorithm_runner_still_reports_its_type(self):
        from restapi.operationworker.taskrunners.taskrunnerprediction import TaskRunnerPrediction, Settings
        from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
        from omr.steps.algorithmtypes import AlgorithmTypes

        runner = TaskRunnerPrediction(AlgorithmTypes.STAFF_LINES_PC,
                                      PageSelection.from_book(DatabaseBook('demo')),
                                      Settings(params=AlgorithmPredictorParams(), store_to_pcgts=False))
        self.assertEqual(runner.operation(), AlgorithmTypes.STAFF_LINES_PC.value)

    def test_the_task_list_survives_a_queued_export(self):
        from unittest import mock
        from restapi.operationworker.taskrunners.taskrunnerdocumentsexport import TaskRunnerDocumentsExport
        from restapi.operationworker.task import Task, TaskStatus
        from rest_framework.test import APIClient

        book = DatabaseBook('demo')
        admin = User.objects.create_superuser('task_list_admin', password='pw')
        task = Task(task_id='export-task',
                    task_runner=TaskRunnerDocumentsExport(book, TaskRunnerDocumentsExport.FORMAT_MEI4_ZIP),
                    task_status=TaskStatus(),
                    task_result={},
                    creator=admin)

        client = APIClient()
        client.force_authenticate(user=admin)
        with mock.patch('restapi.operationworker.operationworker.operation_worker.queue.tasks', [task]):
            response = client.get('/api/tasks')
            self.assertEqual(response.status_code, 200, response.content)
            self.assertEqual([t['algorithmType'] for t in response.json()], ['documents_export'])

            response = client.get('/api/book/demo/tasks')
            self.assertEqual(response.status_code, 200, response.content)
            self.assertEqual([t['algorithmType'] for t in response.json()], ['documents_export'])
