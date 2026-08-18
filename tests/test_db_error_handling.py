import json
import os
import shutil
from io import StringIO
from unittest import mock

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

from django.contrib.auth.models import User  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.db import OperationalError  # noqa: E402
from django.urls import reverse  # noqa: E402
from rest_framework import status  # noqa: E402
from rest_framework.test import APITestCase, APITransactionTestCase  # noqa: E402

from database import DatabaseBook  # noqa: E402
from database.db_errors import describe_db_error, retry_on_db_error  # noqa: E402
from restapi.models.error import ErrorCodes  # noqa: E402

BOOK = 'db_error_test'
PAGES = ['page00000001', 'page00000002']


def io_error():
    """A Django OperationalError wrapping a real sqlite3 error, as the driver raises it."""
    import sqlite3
    try:
        raise sqlite3.OperationalError('disk I/O error')
    except sqlite3.OperationalError as cause:
        return OperationalError('disk I/O error').with_traceback(None) if cause is None \
            else _wrap(cause)


def _wrap(cause):
    error = OperationalError('disk I/O error')
    error.__cause__ = cause
    return error


class DatabaseErrorHandlingTestCase(APITestCase):
    """A transient SQLite failure must heal by reconnecting, and a persistent one must
    reach the client as a 503 it can act on -- never as a 500 that silently costs the
    user the edit lock (and with it the whole tool bar)."""

    def setUp(self):
        self.root = os.path.join(settings.PRIVATE_MEDIA_ROOT, BOOK)
        shutil.rmtree(self.root, ignore_errors=True)
        for page in PAGES:
            os.makedirs(os.path.join(self.root, 'pages', page))
        self.book = DatabaseBook(BOOK)

        User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': 'user', 'password': 'user'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer {0}'.format(response.data['access']))

        self.lock_url = '/api/book/{}/page/{}/lock'.format(BOOK, PAGES[0])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # helpers ------------------------------------------------------------------

    def _patch_lock_query(self, side_effect):
        """Make the PageEditLock lookup in DatabasePage._edit_lock_once fail."""
        from database.models.book_index import PageEditLock
        manager = mock.MagicMock()
        manager.filter.side_effect = side_effect
        return mock.patch.object(PageEditLock, 'objects', manager)

    # api level ----------------------------------------------------------------

    def test_persistent_error_is_a_503_not_a_500(self):
        with self._patch_lock_query(lambda *a, **kw: _raise()):
            for call in (lambda: self.client.get(self.lock_url, format='json'),
                         lambda: self.client.put(self.lock_url, {}, format='json')):
                response = call()
                self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE,
                                 response.content)
                self.assertEqual(json.loads(response.content)['errorCode'],
                                 ErrorCodes.SERVER_DATABASE_UNAVAILABLE.value)

    def test_save_through_require_lock_reports_the_database(self):
        # require_lock fronts every page save: a database failure must not look like a
        # rejected save
        with self._patch_lock_query(lambda *a, **kw: _raise()):
            response = self.client.put(
                '/api/book/{}/page/{}/content/page_progress'.format(BOOK, PAGES[0]),
                {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE, response.content)
        self.assertEqual(json.loads(response.content)['errorCode'],
                         ErrorCodes.SERVER_DATABASE_UNAVAILABLE.value)

    def test_assignments_still_render_without_the_page_index(self):
        with mock.patch('database.book_index.stored_book_pages', side_effect=_raise):
            response = self.client.get('/api/book/{}/assignments'.format(BOOK), format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        body = json.loads(response.content)
        self.assertListEqual(body['pageOrder'], PAGES, 'the page order comes from disk')
        self.assertEqual(body['totalPages'], len(PAGES))

    # index writes -------------------------------------------------------------

    def test_assignments_get_does_not_write_the_index_without_sync(self):
        from database.book_index import index_book
        from database.models.book_index import PageIndex
        index_book(self.book)
        before = list(PageIndex.objects.filter(book__name=BOOK).order_by('name')
                      .values('name', 'pcgts_mtime', 'progress_mtime'))

        # a changed page would normally make sync_book_pages rewrite the row
        with open(os.path.join(self.root, 'pages', PAGES[0], 'page_progress.json'), 'w') as f:
            json.dump({'locked': {'StaffLines': True}, 'verified': False}, f)

        response = self.client.get('/api/book/{}/assignments'.format(BOOK), format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertListEqual(before, list(PageIndex.objects.filter(book__name=BOOK)
                                          .order_by('name')
                                          .values('name', 'pcgts_mtime', 'progress_mtime')),
                             'the default read path must not write to the index')

        response = self.client.get('/api/book/{}/assignments?sync=1'.format(BOOK), format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertNotEqual(before, list(PageIndex.objects.filter(book__name=BOOK)
                                         .order_by('name')
                                         .values('name', 'pcgts_mtime', 'progress_mtime')),
                            '?sync=1 must refresh the index')

    # write contention ---------------------------------------------------------

    def test_connections_are_persistent(self):
        # closing the connection after every request tears the WAL down and rebuilds it
        # constantly; a connection opening into that teardown fails with
        # SQLITE_IOERR_SHORT_READ, so the connections must outlive the request
        from django.conf import settings as django_settings
        self.assertGreater(django_settings.DATABASES['default'].get('CONN_MAX_AGE', 0), 0)
        self.assertTrue(django_settings.DATABASES['default'].get('CONN_HEALTH_CHECKS'))

    def test_wal_pragma_keeps_synchronous_full_when_not_in_wal(self):
        # synchronous=NORMAL is only safe in WAL mode; if the conversion failed the
        # database must keep the stricter setting
        from unittest import mock
        from database.apps import set_sqlite_pragmas

        executed = []

        class FakeCursor:
            def execute(self, sql):
                executed.append(sql)

            def fetchone(self):
                return ('delete',)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        fake_connection = mock.MagicMock(vendor='sqlite')
        fake_connection.cursor.return_value = FakeCursor()
        set_sqlite_pragmas(None, fake_connection)
        self.assertIn('PRAGMA journal_mode=WAL;', executed)
        self.assertNotIn('PRAGMA synchronous=NORMAL;', executed)

    def test_transactions_begin_immediate(self):
        # a DEFERRED transaction that has to upgrade to a writer fails instantly with
        # "database is locked" and the busy timeout cannot help -- so every write
        # transaction must take the lock up front
        from django.db import connection
        self.assertEqual(getattr(connection, 'transaction_mode', None), 'IMMEDIATE')

    # corruption ---------------------------------------------------------------

    def test_corruption_is_reported_with_the_recovery_command(self):
        import logging
        from database.db_errors import is_corruption, log_db_failure
        malformed = _wrap_malformed()
        self.assertTrue(is_corruption(malformed))

        logger = logging.getLogger('tests.corruption')
        with self.assertLogs(logger, level='ERROR') as captured:
            log_db_failure(logger, 'Reading the page index', malformed)
        self.assertIn('DAMAGED', captured.output[0])
        self.assertIn('db_recover', captured.output[0])

    def test_recover_salvages_a_corrupted_database(self):
        import sqlite3
        import tempfile
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'db.sqlite3')
        try:
            connection = sqlite3.connect(path)
            connection.execute('create table users (id integer primary key, name text)')
            connection.execute('create table pages (id integer primary key, blob text)')
            connection.executemany('insert into users (name) values (?)',
                                   [('user{}'.format(i),) for i in range(50)])
            connection.executemany('insert into pages (blob) values (?)',
                                   [('x' * 500,) for _ in range(500)])
            connection.commit()
            connection.close()

            with open(path, 'r+b') as f:   # scribble over a page in the middle
                f.seek(os.path.getsize(path) // 2)
                f.write(b'\xff' * 2048)
            with self.assertRaises(sqlite3.DatabaseError):
                sqlite3.connect(path).execute('select count(*) from pages').fetchone()

            out = StringIO()
            call_command('db_recover', '--path', path, stdout=out)

            recovered = path + '.recovered'
            self.assertTrue(os.path.exists(path), 'the damaged original must be kept')
            self.assertTrue(os.path.exists(recovered))
            check = [row[0] for row in
                     sqlite3.connect(recovered).execute('PRAGMA integrity_check;')]
            self.assertEqual(check, ['ok'])
            users = sqlite3.connect(recovered).execute('select count(*) from users').fetchone()[0]
            self.assertEqual(users, 50, 'the non-rebuildable data must survive')
            self.assertIn('integrity_check', out.getvalue())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_python_salvage_keeps_text_as_text(self):
        """A salvaged database must hold str, not bytes, in its text columns.

        The fallback reads the damaged file with a per-value text factory; a blanket
        ``text_factory = bytes`` used to turn every recovered string into a BLOB, after
        which ``user.username`` came back as b'admin'."""
        import sqlite3
        import tempfile
        from database.management.commands.db_recover import Command
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'db.sqlite3')
        output = os.path.join(directory, 'recovered.sqlite3')
        try:
            connection = sqlite3.connect(path)
            connection.execute('create table users (id integer primary key, name varchar(150))')
            connection.execute("insert into users (name) values ('admin'), ('Bärbel')")
            connection.commit()
            connection.close()

            Command()._recover_with_python(path, output)

            rows = list(sqlite3.connect(output).execute(
                'select typeof(name), name from users order by id'))
            self.assertEqual(rows, [('text', 'admin'), ('text', 'Bärbel')])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_repair_text_converts_blobs_back(self):
        import sqlite3
        import tempfile
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'db.sqlite3')
        try:
            connection = sqlite3.connect(path)
            connection.execute('create table users (id integer primary key, name varchar(150))')
            connection.execute('create table files (id integer primary key, payload blob)')
            connection.execute('insert into users (name) values (?), (?)',
                               (b'admin', 'already-text'))
            connection.execute('insert into files (payload) values (?)', (b'\x00\xff',))
            connection.commit()
            connection.close()

            out = StringIO()
            call_command('db_recover', '--path', path, '--repair-text', stdout=out)

            recovered = sqlite3.connect(path)
            self.assertEqual(
                list(recovered.execute('select typeof(name), name from users order by id')),
                [('text', 'admin'), ('text', 'already-text')])
            self.assertEqual(
                recovered.execute('select typeof(payload) from files').fetchone()[0], 'blob',
                'a column declared blob must be left alone')
            recovered.close()

            backups = [f for f in os.listdir(directory) if '.before-text-repair-' in f]
            self.assertEqual(len(backups), 1, 'the database before the repair must be kept')
            self.assertIn('users.name', out.getvalue())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_repair_text_drops_rows_shadowed_by_a_correct_one(self):
        """A server that kept running after the salvage inserted text rows beside the blob
        ones (SQLite keys b'demo' and 'demo' apart). The blob duplicates must go."""
        import sqlite3
        import tempfile
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'db.sqlite3')
        try:
            connection = sqlite3.connect(path)
            connection.execute('create table books (name varchar(255) primary key, '
                               'display varchar(255))')
            connection.execute('create table pages (id integer primary key, '
                               'book_id varchar(255), name varchar(255))')
            connection.execute('create unique index pages_uniq on pages (book_id, name)')
            connection.execute('insert into books values (?, ?), (?, ?), (?, ?)',
                               (b'demo', 'stale', 'demo', 'live', b'only-blob', 'x'))
            connection.execute('insert into pages (book_id, name) values (?, ?), (?, ?)',
                               (b'demo', '001', 'demo', '001'))
            connection.commit()
            connection.close()

            out = StringIO()
            call_command('db_recover', '--path', path, '--repair-text', stdout=out)

            repaired = sqlite3.connect(path)
            self.assertEqual(
                sorted(repaired.execute('select typeof(name), name, display from books')),
                [('text', 'demo', 'live'), ('text', 'only-blob', 'x')],
                'the live row wins, a blob row without a twin is converted')
            self.assertEqual(
                list(repaired.execute('select typeof(book_id), book_id, name from pages')),
                [('text', 'demo', '001')])
            repaired.close()
            self.assertIn('duplicate row(s) dropped', out.getvalue())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    # management command -------------------------------------------------------

    def test_db_health_runs(self):
        out = StringIO()
        call_command('db_health', '--quick', stdout=out)
        report = out.getvalue()
        self.assertIn('database:', report)
        self.assertIn('PRAGMA quick_check', report)
        self.assertIn('transaction mode:    IMMEDIATE', report)
        self.assertIn('PRAGMA journal_mode', report)
        self.assertIn('ok', report)


def _sqlite_error():
    import sqlite3
    try:
        # a real driver error carries the extended result code describe_db_error reports
        sqlite3.connect('/proc/version/nope/db.sqlite3').execute('select 1')
    except sqlite3.Error as e:
        return e
    return sqlite3.OperationalError('disk I/O error')


class DatabaseRetryTestCase(APITransactionTestCase):
    """Retries need a case that does not wrap each test in a transaction: inside an open
    atomic block retry_on_db_error deliberately re-raises instead of reconnecting."""

    def setUp(self):
        self.root = os.path.join(settings.PRIVATE_MEDIA_ROOT, BOOK)
        shutil.rmtree(self.root, ignore_errors=True)
        for page in PAGES:
            os.makedirs(os.path.join(self.root, 'pages', page))

        User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': 'user', 'password': 'user'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer {0}'.format(response.data['access']))
        self.lock_url = '/api/book/{}/page/{}/lock'.format(BOOK, PAGES[0])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _patch_lock_query(self, side_effect):
        from database.models.book_index import PageEditLock
        manager = mock.MagicMock()
        manager.filter.side_effect = side_effect
        return mock.patch.object(PageEditLock, 'objects', manager)

    def test_describe_db_error_names_the_extended_code(self):
        described = describe_db_error(_wrap(_sqlite_error()))
        self.assertIn('SQLITE_', described)
        self.assertIn('db=', described)
        self.assertIn('free=', described)
        self.assertIn('fds=', described)

    def test_retry_reconnects_and_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise _wrap(_sqlite_error())
            return 'ok'

        self.assertEqual(retry_on_db_error(flaky), 'ok')
        self.assertEqual(len(calls), 2, 'the first failure must be retried once')

    def test_retry_gives_up_after_the_second_failure(self):
        calls = []

        def broken():
            calls.append(1)
            raise _wrap(_sqlite_error())

        with self.assertRaises(OperationalError):
            retry_on_db_error(broken)
        self.assertEqual(len(calls), 2)

    # api level ----------------------------------------------------------------

    def test_transient_error_is_invisible_to_the_client(self):
        from database.models.book_index import PageEditLock
        real_objects = PageEditLock.objects
        state = {'failed': False}

        def once(*args, **kwargs):
            if not state['failed']:
                state['failed'] = True
                raise _wrap(_sqlite_error())
            return real_objects.filter(*args, **kwargs)

        with self._patch_lock_query(once):
            response = self.client.get(self.lock_url, format='json')
        self.assertTrue(state['failed'], 'the patched query never ran')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertFalse(json.loads(response.content)['locked'])

    def test_index_write_retries_on_lock_contention(self):
        from database.book_index import safe
        calls = []

        def flaky_index_update():
            calls.append(1)
            if len(calls) == 1:
                raise _wrap_locked()
            return 'indexed'

        self.assertEqual(safe(flaky_index_update)(), 'indexed')
        self.assertEqual(len(calls), 2)

    def test_index_write_gives_up_quietly(self):
        from database.book_index import safe

        def always_locked():
            raise _wrap_locked()

        # swallowed, as an index row is a cache the next read repairs
        self.assertIsNone(safe(always_locked)())

    def test_no_reconnect_inside_an_atomic_block(self):
        from django.db import transaction
        calls = []

        def failing():
            calls.append(1)
            raise _wrap(_sqlite_error())

        with transaction.atomic():
            with self.assertRaises(OperationalError):
                retry_on_db_error(failing)
        self.assertEqual(len(calls), 1, 'an open transaction must not be retried')



def _wrap_malformed():
    import sqlite3
    error = OperationalError('database disk image is malformed')
    error.__cause__ = sqlite3.DatabaseError('database disk image is malformed')
    return error


def _wrap_locked():
    import sqlite3
    return _wrap(sqlite3.OperationalError('database is locked'))


def _raise(*args, **kwargs):
    raise _wrap(_sqlite_error())
