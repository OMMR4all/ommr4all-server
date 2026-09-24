import json
import os
import shutil
import threading
from unittest.mock import patch

import numpy as np
from PIL import Image

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

from datetime import timedelta  # noqa: E402

from django.contrib.auth.models import User  # noqa: E402
from django.urls import reverse  # noqa: E402
from django.utils import timezone  # noqa: E402
from rest_framework import status  # noqa: E402
from rest_framework.test import APITestCase  # noqa: E402

from database import DatabaseBook  # noqa: E402
from database.database_book_assignments import BookAssignments, DatabaseBookAssignments, \
    PageAssignment  # noqa: E402
from database.database_permissions import BookPermissionFlags, DatabaseBookPermissionFlag  # noqa: E402
from database.models.book_index import PageEditLock  # noqa: E402
from restapi.models.error import ErrorCodes  # noqa: E402
from restapi.operationworker.task import Task, TaskStatus, TaskStatusCodes
from restapi.operationworker.taskqueue import TaskQueue
from restapi.operationworker.taskresources import Resources, TaskResource
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.operationworker.taskrunners.taskrunnersuggestedassignment import TaskRunnerSuggestedAssignment
from restapi.operationworker.operationworker import OperationWorker

BOOK = 'assignments_test'
PAGES = ['page00000001', 'page00000002', 'page00000003', 'page00000004']


class PageAssignmentsTestCase(APITestCase):
    """Page assignments live in storage/<book>/page_assignments.json; progress and
    'currently editing' are derived from the page index, never stored."""

    def setUp(self):
        # a scratch book of empty page folders: these tests rename and delete pages,
        # which must not touch the shared 'demo' fixture
        self.root = os.path.join(settings.PRIVATE_MEDIA_ROOT, BOOK)
        shutil.rmtree(self.root, ignore_errors=True)
        for page in PAGES:
            os.makedirs(os.path.join(self.root, 'pages', page))
            self._write_progress(page, [])
        self.book = DatabaseBook(BOOK)

        User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        User.objects.create_user(username='assignee', email='assignee@mail.com', password='assignee')
        self.admin_auth = self._login('user', 'user')
        self.client.credentials(HTTP_AUTHORIZATION=self.admin_auth)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # helpers ------------------------------------------------------------------

    def _login(self, username, password):
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': username, 'password': password}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        return 'Bearer {0}'.format(response.data['access'])

    def _write_progress(self, page, locks, verified=False):
        all_locks = ['StaffLines', 'Layout', 'Symbols', 'Text']
        with open(os.path.join(self.root, 'pages', page, 'page_progress.json'), 'w') as f:
            json.dump({'locked': {label: (label in locks) for label in all_locks},
                       'verified': verified}, f)

    def _create(self, username='assignee', pages=None, note='a note'):
        response = self.client.put('/api/book/{}/assignments'.format(BOOK),
                                   {'username': username,
                                    'pages': PAGES[:2] if pages is None else pages,
                                    'note': note}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED, response.content)
        return json.loads(response.content)

    def _get(self, sync=False):
        # the overview asks for sync=1 (fresh progress); the highlight consumers use the
        # write-free default
        url = '/api/book/{}/assignments{}'.format(BOOK, '?sync=1' if sync else '')
        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        return json.loads(response.content)

    def _stored_pages(self, id):
        return DatabaseBookAssignments.load(self.book).by_id(id).pages

    # tests --------------------------------------------------------------------

    def test_get_empty_book_without_file(self):
        body = self._get()
        self.assertListEqual(body['assignments'], [])
        self.assertListEqual(body['pageOrder'], PAGES)
        self.assertEqual(body['totalPages'], len(PAGES))
        self.assertEqual(body['assignedPages'], 0)
        self.assertFalse(os.path.exists(DatabaseBookAssignments.path(self.book)))

    def test_create_read_update_delete(self):
        created = self._create()
        self.assertEqual(created['user']['username'], 'assignee')
        self.assertListEqual(created['pages'], PAGES[:2])
        self.assertTrue(os.path.exists(DatabaseBookAssignments.path(self.book)))

        body = self._get()
        self.assertEqual(len(body['assignments']), 1)
        self.assertEqual(body['assignedPages'], 2)

        response = self.client.post('/api/book/{}/assignment/{}'.format(BOOK, created['id']),
                                    {'username': 'assignee', 'pages': PAGES, 'note': 'changed'},
                                    format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        updated = json.loads(response.content)
        self.assertEqual(updated['id'], created['id'], 'the id must survive an update')
        self.assertEqual(updated['note'], 'changed')
        self.assertIsNotNone(updated['updated'])
        self.assertEqual(updated['updatedBy'], 'user')
        self.assertListEqual(self._stored_pages(created['id']), PAGES)

        response = self.client.delete('/api/book/{}/assignment/{}'.format(BOOK, created['id']))
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertListEqual(self._get()['assignments'], [])

    def test_pages_are_deduped_and_sorted_into_book_order(self):
        created = self._create(pages=[PAGES[2], PAGES[0], PAGES[0]])
        self.assertListEqual(created['pages'], [PAGES[0], PAGES[2]])

    def test_unknown_assignment_is_rejected(self):
        for call in (lambda: self.client.post('/api/book/{}/assignment/{}'.format(BOOK, 'abc123'),
                                              {'username': 'assignee', 'pages': [], 'note': ''}, format='json'),
                     lambda: self.client.delete('/api/book/{}/assignment/{}'.format(BOOK, 'abc123'))):
            response = call()
            self.assertEqual(response.status_code, status.HTTP_406_NOT_ACCEPTABLE, response.content)
            self.assertEqual(json.loads(response.content)['errorCode'],
                             ErrorCodes.BOOK_ASSIGNMENT_NOT_FOUND.value)

    def test_rejects_unknown_user_and_unknown_page(self):
        response = self.client.put('/api/book/{}/assignments'.format(BOOK),
                                   {'username': 'nobody', 'pages': PAGES[:1], 'note': ''}, format='json')
        self.assertEqual(response.status_code, status.HTTP_406_NOT_ACCEPTABLE, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_ASSIGNMENT_UNKNOWN_USER.value)

        response = self.client.put('/api/book/{}/assignments'.format(BOOK),
                                   {'username': 'assignee', 'pages': ['no_such_page'], 'note': ''},
                                   format='json')
        self.assertEqual(response.status_code, status.HTTP_406_NOT_ACCEPTABLE, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.BOOK_ASSIGNMENT_UNKNOWN_PAGE.value)

    def test_reader_may_view_but_not_write(self):
        self._create()
        User.objects.create_user(username='reader', email='reader@mail.com', password='reader')
        self.book.get_permissions().get_or_add_user_permissions(
            'reader', BookPermissionFlags(DatabaseBookPermissionFlag.READ))
        self.client.credentials(HTTP_AUTHORIZATION=self._login('reader', 'reader'))

        body = self._get()
        self.assertEqual(len(body['assignments']), 1)
        self.assertFalse(BookPermissionFlags(body['permissions']).has(
            DatabaseBookPermissionFlag.EDIT_PERMISSIONS))

        response = self.client.put('/api/book/{}/assignments'.format(BOOK),
                                   {'username': 'reader', 'pages': [], 'note': ''}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)

    def test_assignee_without_access_is_flagged(self):
        created = self._create()
        self.assertFalse(created['userHasAccess'])
        self.book.get_permissions().get_or_add_user_permissions(
            'assignee', BookPermissionFlags(DatabaseBookPermissionFlag.READ))
        self.assertTrue(self._get()['assignments'][0]['userHasAccess'])

    def test_deleted_user_is_reported_as_missing(self):
        self._create()
        User.objects.get(username='assignee').delete()
        assignment = self._get()['assignments'][0]
        self.assertFalse(assignment['userExists'])
        self.assertEqual(assignment['user']['username'], 'assignee')

    def test_progress_is_derived_from_page_progress(self):
        self._write_progress(PAGES[0], ['StaffLines', 'Layout', 'Symbols', 'Text'], verified=True)
        self._write_progress(PAGES[1], ['StaffLines', 'Layout'])
        self._write_progress(PAGES[2], [])
        self._create(pages=PAGES[:3])

        progress = self._get(sync=True)['assignments'][0]['progress']
        self.assertEqual(progress['total'], 3)
        self.assertEqual(progress['existing'], 3)
        self.assertEqual(progress['missing'], 0)
        self.assertEqual(progress['finished'], 1)
        self.assertEqual(progress['inProgress'], 1)
        self.assertEqual(progress['untouched'], 1)
        self.assertEqual(progress['verified'], 1)
        self.assertEqual(progress['locks'], {'StaffLines': 2, 'Layout': 2, 'Symbols': 1, 'Text': 1})

    def test_progress_without_sync_serves_the_stored_index(self):
        # the default read path must not stat and re-index the whole book on every call:
        # pages that exist but have no index row yet count as untouched, not as missing
        self._write_progress(PAGES[0], ['StaffLines', 'Layout', 'Symbols', 'Text'], verified=True)
        self._create(pages=PAGES[:2])

        progress = self._get()['assignments'][0]['progress']
        self.assertEqual(progress['existing'], 2)
        self.assertEqual(progress['missing'], 0)
        self.assertEqual(progress['untouched'], 2)

        self.assertEqual(self._get(sync=True)['assignments'][0]['progress']['finished'], 1)

    def test_currently_editing_reports_live_edit_locks(self):
        self._create(pages=PAGES)
        self.book.page(PAGES[1]).lock(User.objects.get(username='assignee'))
        editing = self._get()['currentlyEditing']
        self.assertEqual(len(editing), 1)
        self.assertEqual(editing[0]['page'], PAGES[1])
        self.assertEqual(editing[0]['user']['username'], 'assignee')

        # a stale lock is not reported (it is expired on the page path, not here)
        ttl = getattr(settings, 'PAGE_EDIT_LOCK_TTL_HOURS', 12)
        PageEditLock.objects.filter(page__book__name=BOOK).update(
            acquired_at=timezone.now() - timedelta(hours=ttl + 1))
        self.assertListEqual(self._get()['currentlyEditing'], [])

    def test_currently_editing_without_any_assignment(self):
        # "who is working on which page" must answer even for a book nobody was assigned to
        self.book.page(PAGES[0]).lock(User.objects.get(username='assignee'))
        body = self._get()
        self.assertListEqual(body['assignments'], [])
        self.assertEqual(len(body['currentlyEditing']), 1)
        self.assertEqual(body['currentlyEditing'][0]['page'], PAGES[0])
        self.assertEqual(body['currentlyEditing'][0]['user']['username'], 'assignee')
        self.assertFalse(os.path.exists(DatabaseBookAssignments.path(self.book)))

    def test_page_rename_follows_the_assignment(self):
        created = self._create(pages=PAGES[:2])
        response = self.client.post('/api/book/{}/page/{}/rename'.format(BOOK, PAGES[0]),
                                    {'name': 'renamed_page'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertIn('renamed_page', self._stored_pages(created['id']))
        self.assertNotIn(PAGES[0], self._stored_pages(created['id']))

    def test_bulk_rename_follows_the_assignment(self):
        created = self._create(pages=PAGES[:2])
        files = [{'src': src, 'target': target} for src, target in
                 zip(PAGES, ['renamed_{}'.format(i) for i in range(len(PAGES))])]
        response = self.client.post('/api/book/{}/rename_pages/'.format(BOOK),
                                    {'files': files}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertListEqual(self._stored_pages(created['id']), ['renamed_0', 'renamed_1'])

    def test_page_delete_drops_the_label(self):
        created = self._create(pages=PAGES[:2])
        self.book.page(PAGES[0]).delete()
        self.assertListEqual(self._stored_pages(created['id']), [PAGES[1]])

    def test_missing_pages_are_reported_not_pruned(self):
        created = self._create(pages=PAGES[:2])
        # a page vanishing out of band (e.g. an imported book) must stay visible
        shutil.rmtree(os.path.join(self.root, 'pages', PAGES[0]))
        assignment = self._get()['assignments'][0]
        self.assertEqual(assignment['progress']['missing'], 1)
        self.assertEqual(assignment['progress']['existing'], 1)
        self.assertListEqual(self._stored_pages(created['id']), PAGES[:2])

    def test_concurrent_mutations_do_not_lose_an_assignment(self):
        errors = []

        def add(name):
            def apply(assignments: BookAssignments):
                assignments.assignments.append(PageAssignment(id=name, username='assignee'))
            try:
                DatabaseBookAssignments.mutate(self.book, apply)
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=add, args=('id{}'.format(i),)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertListEqual(errors, [])
        stored = DatabaseBookAssignments.load(self.book)
        self.assertEqual(len(stored.assignments), 8,
                         [a.id for a in stored.assignments])

    def test_suggested_worker_excludes_finished_pages_and_rechecks_assignments(self):
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 50, 100, 180)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        self._write_progress(PAGES[0], ['Symbols'])
        self._write_progress(PAGES[1], ['StaffLines', 'Layout', 'Symbols', 'Text'])
        runner = TaskRunnerSuggestedAssignment(
            self.book, 'assignee', 2,
            Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)]))
        user = User.objects.get(username='assignee')
        task = Task('test-run', runner, TaskStatus(TaskStatusCodes.RUNNING), {}, user)

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                value = float(pixels[0, 0, 0]) / 255.0
                features = np.array([[[value, 1.0 - value]]], dtype=np.float32)
                return SpatialFeatureMap(features, 32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        messages = Messages()
        with patch('restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                   'build_feature_extractor', return_value=Extractor()) as factory:
            result = runner.run(task, messages)
        self.assertEqual(factory.call_args.args[0].max_input_side, 448)
        assignment = DatabaseBookAssignments.load(self.book).by_id(result['assignment_id'])
        self.assertEqual(len(assignment.pages), 2)
        self.assertNotIn(PAGES[1], assignment.pages)
        self.assertEqual(messages.items[-1].status.n_processed, 4)

        # A second user's selection must re-read assignments under the mutation lock.
        other = TaskRunnerSuggestedAssignment(
            self.book, 'user', 2, Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)]))
        second = Task('test-other', other, TaskStatus(TaskStatusCodes.RUNNING), {},
                      User.objects.get(username='user'))
        with patch('restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                   'build_feature_extractor', return_value=Extractor()):
            result = other.run(second, Messages())
        self.assertEqual(result, {'error': 'insufficient_pages', 'available': 1,
                                  'excluded_pages': []})
        self.assertEqual(len(DatabaseBookAssignments.load(self.book).assignments), 1)

    def test_unique_queue_reuses_same_user_task_across_counts(self):
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        first = TaskRunnerSuggestedAssignment(self.book, 'assignee', 1, resources)
        second = TaskRunnerSuggestedAssignment(self.book, 'assignee', 3, resources)
        queue = TaskQueue()
        creator = User.objects.get(username='assignee')
        self.assertEqual(queue.put_unique('first', first, creator), ('first', True))
        self.assertEqual(queue.put_unique('second', second, creator), ('first', False))
        self.assertEqual(queue.task_for_id('first').creator.username, 'assignee')

    def test_self_assignment_api_auth_idempotency_and_atomic_insufficient(self):
        from restapi.views import bookassignments as view
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        self.book.get_permissions().get_or_add_user_permissions(
            'assignee', BookPermissionFlags(DatabaseBookPermissionFlag.READ_WRITE))
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 60, 80, 150)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        self._write_progress(PAGES[3], ['StaffLines', 'Layout', 'Symbols', 'Text'])
        base = '/api/book/{}/assignments/self'.format(BOOK)

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                value = float(pixels[0, 0, 0]) / 255.0
                return SpatialFeatureMap(
                    np.array([[[value, 1.0 - value]]], dtype=np.float32),
                    32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def put(self, message):
                pass

        with patch.object(view, 'operation_worker', worker), patch(
                'restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                'build_feature_extractor', return_value=Extractor()):
            for invalid in (True, 0, -1, 1.5, '2', 5):
                response = self.client.put(base, {'count': invalid}, format='json')
                self.assertEqual(response.status_code, 400, response.content)
                self.assertEqual(response.data['errorCode'],
                                 ErrorCodes.BOOK_ASSIGNMENT_INVALID_COUNT.value)
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            self.client.credentials(HTTP_AUTHORIZATION=self._login('assignee', 'assignee'))
            first = self.client.put(base, {'count': 2, 'username': 'user',
                                           'pages': [PAGES[3]]}, format='json')
            self.assertEqual(first.status_code, 202, first.content)
            task_id = first.data['task_id']
            self.assertEqual(self.client.put(base, {'count': 3}, format='json').data['task_id'],
                             task_id)
            self.assertEqual(self.client.get(base).data['task_id'], task_id)
            task_url = base + '/task/' + task_id
            self.assertEqual(self.client.get(task_url).data['status']['code'],
                             TaskStatusCodes.QUEUED.value)
            self.client.credentials(HTTP_AUTHORIZATION=self.admin_auth)
            self.assertEqual(self.client.get(task_url).status_code, 406)
            self.client.credentials(HTTP_AUTHORIZATION=self._login('assignee', 'assignee'))
            task = worker.queue.task_for_id(task_id)
            result = task.task_runner.run(task, Messages())
            worker.queue.update_status(task_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            finished = self.client.get(task_url)
            self.assertEqual(finished.status_code, 200, finished.content)
            self.assertEqual(finished.data['assignment']['user']['username'], 'assignee')
            self.assertEqual(len(finished.data['assignment']['pages']), 2)
            self.assertNotIn(PAGES[3], finished.data['assignment']['pages'])
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            self.assertEqual(self.client.get(task_url).status_code, 406)

            second = self.client.put(base, {'count': 2}, format='json').data['task_id']
            task = worker.queue.task_for_id(second)
            result = task.task_runner.run(task, Messages())
            worker.queue.update_status(second, TaskStatus(TaskStatusCodes.FINISHED), result)
            failed = self.client.get(base + '/task/' + second)
            self.assertEqual(failed.status_code, 409, failed.content)
            self.assertEqual(failed.data['errorCode'],
                             ErrorCodes.BOOK_ASSIGNMENT_INSUFFICIENT_PAGES.value)
            self.assertIn('1 eligible', failed.data['userMessage'])
            self.assertEqual(len(DatabaseBookAssignments.load(self.book).assignments), 1)
        worker.shutdown()

    def test_suggested_range_limits_embedding_and_atomic_selection(self):
        from restapi.views import bookassignments as view
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 60, 80, 150)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        self._write_progress(PAGES[1], ['Symbols'])
        self.book.get_permissions().get_or_add_user_permissions(
            'assignee', BookPermissionFlags(DatabaseBookPermissionFlag.READ_WRITE))
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        self.client.credentials(HTTP_AUTHORIZATION=self._login('assignee', 'assignee'))
        base = '/api/book/{}/assignments/self'.format(BOOK)

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                value = float(pixels[0, 0, 0]) / 255.0
                return SpatialFeatureMap(np.array([[[value, 1.0 - value]]], dtype=np.float32),
                                         32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        with patch.object(view, 'operation_worker', worker), patch(
                'restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                'build_feature_extractor', return_value=Extractor()):
            for invalid in ({'fromPage': PAGES[1]}, {'fromPage': PAGES[1], 'toPage': 'absent'},
                            {'fromPage': 1, 'toPage': PAGES[2]},
                            {'fromPage': None, 'toPage': None}):
                response = self.client.put(base, {'count': 1, **invalid}, format='json')
                self.assertEqual(response.status_code, 400, response.content)
                self.assertEqual(response.data['errorCode'],
                                 ErrorCodes.BOOK_ASSIGNMENT_INVALID_RANGE.value)
            request = {'count': 3, 'fromPage': PAGES[2], 'toPage': PAGES[1]}
            first = self.client.put(base, request, format='json')
            self.assertEqual(first.status_code, 202, first.content)
            first_id = first.data['task_id']
            # The outstanding task keeps its original range even if a second PUT changes it.
            duplicate = self.client.put(base, {'count': 1, 'fromPage': PAGES[0],
                                                'toPage': PAGES[3]}, format='json')
            self.assertEqual(duplicate.data['task_id'], first_id)
            task = worker.queue.task_for_id(first_id)
            self.assertEqual(task.task_runner.range_pages, frozenset(PAGES[1:3]))
            messages = Messages()
            result = task.task_runner.run(task, messages)
            self.assertEqual(messages.items[-1].status.n_total, 2)
            self.assertEqual(result['available'], 2)
            worker.queue.update_status(first_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            failed = self.client.get(base + '/task/' + first_id)
            self.assertEqual(failed.status_code, 409, failed.content)
            self.assertIn('2 eligible', failed.data['userMessage'])
            self.assertFalse(os.path.exists(DatabaseBookAssignments.path(self.book)))

            second = self.client.put(base, {**request, 'count': 2}, format='json')
            self.assertEqual(second.status_code, 202, second.content)
            second_id = second.data['task_id']
            task = worker.queue.task_for_id(second_id)
            result = task.task_runner.run(task, Messages())
            worker.queue.update_status(second_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            finished = self.client.get(base + '/task/' + second_id)
            self.assertEqual(finished.status_code, 200, finished.content)
            self.assertEqual(finished.data['assignment']['pages'], PAGES[1:3])
            self.assertEqual(len(DatabaseBookAssignments.load(self.book).assignments), 1)
        worker.shutdown()


    def test_batch_suggested_assignments_are_disjoint_and_creator_scoped(self):
        from restapi.views import bookassignments as view
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 55, 80, 150)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        User.objects.create_user(username='other', password='other')
        User.objects.create_superuser(username='manager', email='manager@mail.com',
                                      password='manager')
        base = '/api/book/{}/assignments/suggested'.format(BOOK)

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                value = float(pixels[0, 0, 0]) / 255.0
                return SpatialFeatureMap(np.array([[[value, 1.0 - value]]], dtype=np.float32),
                                         32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        with patch.object(view, 'operation_worker', worker), patch(
                'restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                'build_feature_extractor', return_value=Extractor()):
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            request = {'usernames': ['assignee', 'other'], 'count': 2}
            queued = self.client.put(base, request, format='json')
            self.assertEqual(queued.status_code, 202, queued.content)
            task_id = queued.data['task_id']
            self.assertEqual(self.client.get(base).data, {'task_id': task_id})
            self.assertEqual(self.client.get('/api/book/{}/assignments/self'.format(BOOK)).status_code,
                             200)
            self.assertEqual(self.client.get(base + '/task/' + task_id).data['status']['code'],
                             TaskStatusCodes.QUEUED.value)
            self.assertEqual(self.client.put(base, {'usernames': ['other'], 'count': 1,
                                                    'fromPage': PAGES[0],
                                                    'toPage': PAGES[1]},
                                             format='json').data['task_id'], task_id)
            self.client.credentials(HTTP_AUTHORIZATION=self._login('manager', 'manager'))
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            self.assertEqual(self.client.get(base + '/task/' + task_id).status_code, 406)
            self.client.credentials(HTTP_AUTHORIZATION=self.admin_auth)
            task = worker.queue.task_for_id(task_id)
            messages = Messages()
            result = task.task_runner.run(task, messages)
            self.assertEqual(messages.items[-1].status.n_total, 4)
            self.assertEqual(len(result['assignment_ids']), 2)
            worker.queue.update_status(task_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            finished = self.client.get(base + '/task/' + task_id)
            self.assertEqual(finished.status_code, 200, finished.content)
            assignments = finished.data['assignments']
            self.assertEqual([a['user']['username'] for a in assignments], request['usernames'])
            self.assertEqual([len(a['pages']) for a in assignments], [2, 2])
            self.assertEqual(set(assignments[0]['pages']) & set(assignments[1]['pages']), set())
            self.assertEqual(set(assignments[0]['pages']) | set(assignments[1]['pages']),
                             set(PAGES))
            self.assertTrue(all(a['createdBy'] == 'user' for a in assignments))
            self.assertTrue(all(a['userHasAccess'] is False for a in assignments))
            self.assertEqual(len(DatabaseBookAssignments.load(self.book).assignments), 2)
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            self.assertEqual(self.client.get(base + '/task/' + task_id).status_code, 406)
        worker.shutdown()

    def test_batch_suggestion_validates_targets_range_and_permission(self):
        from restapi.views import bookassignments as view
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        base = '/api/book/{}/assignments/suggested'.format(BOOK)
        User.objects.create_user(username='reader', password='reader')
        self.book.get_permissions().get_or_add_user_permissions(
            'reader', BookPermissionFlags(DatabaseBookPermissionFlag.READ))
        with patch.object(view, 'operation_worker', worker):
            for usernames in ([], ['assignee', 'assignee'], ['missing'],
                              ['assignee', 'missing'], [''], 'assignee'):
                response = self.client.put(base, {'usernames': usernames, 'count': 1},
                                           format='json')
                self.assertEqual(response.status_code, 400, response.content)
            for count in (True, 0, -1, 1.5, '1'):
                response = self.client.put(base, {'usernames': ['assignee'], 'count': count},
                                           format='json')
                self.assertEqual(response.status_code, 400, response.content)
            for range_data in ({'fromPage': PAGES[0]}, {'fromPage': 'missing',
                                                         'toPage': PAGES[1]},
                               {'fromPage': None, 'toPage': None}):
                response = self.client.put(base, {'usernames': ['assignee'], 'count': 1,
                                                   **range_data}, format='json')
                self.assertEqual(response.status_code, 400, response.content)
            self.assertEqual(self.client.get(base).data, {'task_id': None})
            task_id = self.client.put(base, {'usernames': ['assignee'], 'count': 1},
                                      format='json').data['task_id']
            self.client.credentials(HTTP_AUTHORIZATION=self._login('reader', 'reader'))
            self.assertEqual(self.client.put(base, {'usernames': ['assignee'], 'count': 1},
                                             format='json').status_code, 401)
            self.assertEqual(self.client.get(base).status_code, 401)
            self.assertEqual(self.client.get(base + '/task/' + task_id).status_code, 401)
            self.client.credentials()
            self.assertIn(self.client.get(base).status_code, (401, 403))
            self.assertIn(self.client.get(base + '/task/' + task_id).status_code, (401, 403))
            self.client.credentials(HTTP_AUTHORIZATION=self.admin_auth)
        worker.shutdown()

    def test_batch_suggestion_insufficient_at_commit_creates_no_partial_assignments(self):
        from restapi.views import bookassignments as view
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 55, 80, 150)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        User.objects.create_user(username='other', password='other')
        base = '/api/book/{}/assignments/suggested'.format(BOOK)

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                return SpatialFeatureMap(np.ones((1, 1, 2), dtype=np.float32),
                                         32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def put(self, message):
                pass

        with patch.object(view, 'operation_worker', worker), patch(
                'restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                'build_feature_extractor', return_value=Extractor()):
            queued = self.client.put(base, {'usernames': ['assignee', 'other'], 'count': 2},
                                     format='json')
            self.assertEqual(queued.status_code, 202, queued.content)
            task_id = queued.data['task_id']
            existing = self._create(username='other', pages=[PAGES[0]])
            task = worker.queue.task_for_id(task_id)
            result = task.task_runner.run(task, Messages())
            self.assertEqual(result['error'], 'insufficient_pages')
            self.assertEqual(result['available'], 3)
            worker.queue.update_status(task_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            failed = self.client.get(base + '/task/' + task_id)
            self.assertEqual(failed.status_code, 409, failed.content)
            self.assertEqual(failed.data['errorCode'],
                             ErrorCodes.BOOK_ASSIGNMENT_INSUFFICIENT_PAGES.value)
            stored = DatabaseBookAssignments.load(self.book).assignments
            self.assertEqual([assignment.id for assignment in stored], [existing['id']])
            # Even a count larger than the book is an asynchronous insufficiency, not a
            # request-shape error; it must preserve the already stored assignment.
            oversized = self.client.put(base, {'usernames': ['assignee', 'other'],
                                                'count': len(PAGES)}, format='json')
            self.assertEqual(oversized.status_code, 202, oversized.content)
            oversized_id = oversized.data['task_id']
            task = worker.queue.task_for_id(oversized_id)
            oversized_result = task.task_runner.run(task, Messages())
            worker.queue.update_status(oversized_id, TaskStatus(TaskStatusCodes.FINISHED),
                                       oversized_result)
            self.assertEqual(self.client.get(base + '/task/' + oversized_id).status_code, 409)
            self.assertEqual([a.id for a in DatabaseBookAssignments.load(self.book).assignments],
                             [existing['id']])
            self._write_progress(PAGES[2], ['StaffLines', 'Layout', 'Symbols', 'Text'])
            ranged = self.client.put(base, {'usernames': ['assignee', 'other'], 'count': 1,
                                            'fromPage': PAGES[3], 'toPage': PAGES[1]},
                                     format='json')
            self.assertEqual(ranged.status_code, 202, ranged.content)
            ranged_id = ranged.data['task_id']
            task = worker.queue.task_for_id(ranged_id)
            self.assertEqual(task.task_runner.range_pages, frozenset(PAGES[1:]))
            result = task.task_runner.run(task, Messages())
            worker.queue.update_status(ranged_id, TaskStatus(TaskStatusCodes.FINISHED), result)
            assignments = self.client.get(base + '/task/' + ranged_id).data['assignments']
            self.assertEqual({page for assignment in assignments for page in assignment['pages']},
                             {PAGES[1], PAGES[3]})
            self.assertTrue(all(len(assignment['pages']) == 1 for assignment in assignments))
        worker.shutdown()

    def test_self_assignment_denies_reader_anonymous_and_foreign_task(self):
        from restapi.views import bookassignments as view
        base = '/api/book/{}/assignments/self'.format(BOOK)
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        worker = OperationWorker(resources=resources, watcher_interval=0)
        worker.task_creator = lambda: None
        worker.health = lambda: {'healthy': True}
        self.book.get_permissions().get_or_add_user_permissions(
            'assignee', BookPermissionFlags(DatabaseBookPermissionFlag.READ_WRITE))
        User.objects.create_user(username='reader', password='reader')
        self.book.get_permissions().get_or_add_user_permissions(
            'reader', BookPermissionFlags(DatabaseBookPermissionFlag.READ))
        with patch.object(view, 'operation_worker', worker):
            self.client.credentials(HTTP_AUTHORIZATION=self._login('assignee', 'assignee'))
            task_id = self.client.put(base, {'count': 1}, format='json').data['task_id']
            self.client.credentials(HTTP_AUTHORIZATION=self._login('reader', 'reader'))
            self.assertEqual(self.client.put(base, {'count': 1}, format='json').status_code, 401)
            self.assertEqual(self.client.get(base).status_code, 401)
            self.assertEqual(self.client.get(base + '/task/' + task_id).status_code, 401)
            self.client.credentials()
            self.assertIn(self.client.get(base).status_code, (401, 403))
            self.client.credentials(HTTP_AUTHORIZATION=self.admin_auth)
            self.assertEqual(self.client.get(base + '/task/' + task_id).status_code, 406)
            self.assertEqual(self.client.get(base + '/task/' + task_id).data['errorCode'],
                             ErrorCodes.BOOK_ASSIGNMENT_NOT_FOUND.value)
        worker.shutdown()

    def test_corrupt_assignments_never_get_overwritten(self):
        path = DatabaseBookAssignments.path(self.book)
        with open(path, 'w') as handle:
            handle.write('{broken JSON')
        with self.assertRaises(Exception):
            DatabaseBookAssignments.mutate(
                self.book, lambda stored: stored.assignments.append(
                    PageAssignment(id='must-not-exist')))
        with open(path) as handle:
            self.assertEqual(handle.read(), '{broken JSON')

    def test_concurrent_suggested_workers_never_share_pages(self):
        for index, name in enumerate(PAGES):
            Image.new('RGB', (19, 23), (index * 55, 80, 150)).save(
                os.path.join(self.root, 'pages', name, 'color_original.jpg'))
        resources = Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)])
        errors = []
        results = []

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                value = float(pixels[0, 0, 0]) / 255.0
                return SpatialFeatureMap(
                    np.array([[[value, 1.0 - value]]], dtype=np.float32),
                    32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def put(self, message):
                pass

        def run(username):
            try:
                runner = TaskRunnerSuggestedAssignment(self.book, username, 2, resources)
                task = Task(username, runner, TaskStatus(TaskStatusCodes.RUNNING), {},
                            User(username=username))
                results.append(runner.run(task, Messages()))
            except Exception as error:
                errors.append(error)

        # Only the assignment mutation is shared; progress reads use fresh disk objects.
        with patch('restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                   'build_feature_extractor', side_effect=lambda config: Extractor()), patch(
                'restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                'prefill_page_progress',
                side_effect=lambda book: [book.page(name) for name in PAGES]):
            threads = [threading.Thread(target=run, args=(username,))
                       for username in ('user', 'assignee')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        stored = DatabaseBookAssignments.load(self.book).assignments
        self.assertEqual(len(stored), 2)
        self.assertEqual(set(stored[0].pages) & set(stored[1].pages), set())
        self.assertEqual(len(set(stored[0].pages) | set(stored[1].pages)), 4)

    def test_unreadable_and_missing_originals_never_create_partial_assignment(self):
        Image.new('RGB', (19, 23), 'white').save(
            os.path.join(self.root, 'pages', PAGES[0], 'color_original.jpg'))
        with open(os.path.join(self.root, 'pages', PAGES[1], 'color_original.jpg'), 'wb') as image:
            image.write(b'not a JPEG')
        runner = TaskRunnerSuggestedAssignment(
            self.book, 'assignee', 2,
            Resources([TaskResource(TaskWorkerGroup.LONG_TASKS_CPU)]))

        class Extractor:
            def load(self):
                pass

            def extract_feature_map(self, pixels, staff_space):
                from omr.discovery.features.base import SpatialFeatureMap
                return SpatialFeatureMap(np.ones((1, 1, 2), dtype=np.float32),
                                         32, (32, 32), (23, 19), (23, 19))

        class Messages:
            def put(self, message):
                pass

        task = Task('missing', runner, TaskStatus(TaskStatusCodes.RUNNING), {},
                    User.objects.get(username='assignee'))
        with patch('restapi.operationworker.taskrunners.taskrunnersuggestedassignment.'
                   'build_feature_extractor', return_value=Extractor()):
            result = runner.run(task, Messages())
        self.assertEqual(result['error'], 'insufficient_pages')
        self.assertEqual(result['available'], 1)
        self.assertEqual({item['reason'] for item in result['excluded_pages']},
                         {'missing_original_image', 'unreadable_original_image'})
        self.assertFalse(os.path.exists(DatabaseBookAssignments.path(self.book)))
