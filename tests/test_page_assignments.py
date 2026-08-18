import json
import os
import shutil
import threading

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
