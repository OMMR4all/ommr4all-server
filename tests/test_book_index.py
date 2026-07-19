import io
import json
import os
import zipfile

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

from django.contrib.auth.models import User  # noqa: E402
from django.urls import reverse  # noqa: E402
from rest_framework import status  # noqa: E402
from rest_framework.test import APITestCase  # noqa: E402

from database import DatabaseBook  # noqa: E402
from database import book_index, pcgts_cache  # noqa: E402
from database.models.book_index import BookIndex, PageIndex, PageEditLock  # noqa: E402


class BookIndexTestCase(APITestCase):
    """The DB index is a rebuildable mirror of the storage folder: these tests cover
    idempotent reindexing, mtime self-healing, DB edit locks and book import."""

    def setUp(self):
        url = reverse('token_obtain_pair')
        User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        response = self.client.post(url, {'username': 'user', 'password': 'user'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer {0}'.format(response.data['access']))
        self.book = DatabaseBook('demo')

    def _page_row_values(self):
        fields = ['name', 'pcgts_mtime', 'progress_mtime', 'has_symbols', 'verified', 'progress_locks']
        return list(PageIndex.objects.filter(book__name='demo').order_by('name').values(*fields))

    def test_reindex_idempotent(self):
        book_index.index_book(self.book)
        first = self._page_row_values()
        self.assertGreater(len(first), 0)
        book_index.index_book(self.book)
        self.assertEqual(first, self._page_row_values())

    def test_prune_removes_vanished_pages(self):
        row = book_index.index_book(self.book)
        PageIndex.objects.create(book=row, name='no_such_page_dir')
        book_index.index_book(self.book)
        self.assertFalse(PageIndex.objects.filter(book=row, name='no_such_page_dir').exists())

    def test_stale_index_self_heals_on_progress_change(self):
        book_index.index_book(self.book)
        page = self.book.page('page00000001')
        progress_path = page.local_file_path('page_progress.json')
        original = None
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                original = f.read()
        try:
            with open(progress_path, 'w') as f:
                json.dump({'locked': {'StaffLines': True, 'Layout': True, 'Symbols': True, 'Text': True},
                           'verified': False}, f)
            rows = {r.name: r for r in book_index.sync_book_pages(self.book)}
            self.assertTrue(all(rows['page00000001'].progress_locks.values()))
        finally:
            if original is None:
                os.remove(progress_path)
            else:
                with open(progress_path, 'w') as f:
                    f.write(original)
            book_index.sync_book_pages(self.book)

    def test_book_counts_cached_per_page(self):
        first = book_index.book_counts(self.book)
        self.assertGreater(first.n_pages, 0)
        # every page now has stored counts matching its pcgts mtime
        for row in PageIndex.objects.filter(book__name='demo'):
            self.assertIsNotNone(row.counts)
            self.assertEqual(row.counts_mtime, row.pcgts_mtime)
        self.assertEqual(first, book_index.book_counts(self.book))

    def test_books_list_backed_by_index(self):
        response = self.client.get('/api/books', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        books = json.loads(response.content)['books']
        self.assertIn('demo', [b['id'] for b in books])
        self.assertTrue(BookIndex.objects.filter(name='demo').exists())

    def test_lock_api_semantics(self):
        page = 'page_test_lock'
        lock_url = '/api/book/demo/page/{}/lock'.format(page)

        # acquire
        response = self.client.put(lock_url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertTrue(json.loads(response.content)['locked'])
        self.assertTrue(PageEditLock.objects.filter(page__book__name='demo', page__name=page).exists())

        # a second user is refused and told who holds the lock, force takes over
        User.objects.create_superuser(username='user2', email='user2@mail.com', password='user2')
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': 'user2', 'password': 'user2'}, format='json')
        client2_auth = 'Bearer {0}'.format(response.data['access'])
        self.client.credentials(HTTP_AUTHORIZATION=client2_auth)

        response = self.client.put(lock_url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        body = json.loads(response.content)
        self.assertFalse(body['locked'])
        self.assertEqual(body['email'], 'user@mail.com')

        response = self.client.put(lock_url, {'force': True}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertTrue(json.loads(response.content)['locked'])
        lock = PageEditLock.objects.get(page__book__name='demo', page__name=page)
        self.assertEqual(lock.user.username, 'user2')

        # release
        response = self.client.delete(lock_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertFalse(PageEditLock.objects.filter(page__book__name='demo', page__name=page).exists())

    def test_deleting_page_row_cascades_lock(self):
        page = 'page_test_lock'
        db_page = self.book.page(page)
        db_page.lock(User.objects.get(username='user'))
        self.assertTrue(db_page.is_locked())
        PageIndex.objects.filter(book__name='demo', name=page).delete()
        self.assertFalse(PageEditLock.objects.filter(page__name=page).exists())

    def test_import_book_is_indexed(self):
        book_name = 'index_import_test'
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w') as zf:
                zf.writestr(book_name + '/book_meta.json', json.dumps({'name': 'Index Import Test'}))
                zf.writestr(book_name + '/pages/', '')
            buf.seek(0)
            buf.name = book_name + '.zip'
            response = self.client.post('/api/books/import', {'file': buf}, format='multipart')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

            self.assertTrue(BookIndex.objects.filter(name=book_name).exists())
            response = self.client.get('/api/books', format='json')
            self.assertIn(book_name, [b['id'] for b in json.loads(response.content)['books']])
        finally:
            DatabaseBook(book_name).delete()
        self.assertFalse(BookIndex.objects.filter(name=book_name).exists())

    def test_pcgts_cache_hit_and_invalidation(self):
        page = self.book.page('page00000001')
        path = page.local_file_path('pcgts.json')
        pcgts_cache.clear()

        first = pcgts_cache.get(self.book.page('page00000001'))
        second = pcgts_cache.get(self.book.page('page00000001'))
        self.assertIs(first, second)

        # a changed mtime invalidates the entry
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        try:
            third = pcgts_cache.get(self.book.page('page00000001'))
            self.assertIsNot(first, third)
        finally:
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        # explicit invalidation drops the entry
        pcgts_cache.invalidate(path)
        fourth = pcgts_cache.get(self.book.page('page00000001'))
        self.assertIsNot(third, fourth)
