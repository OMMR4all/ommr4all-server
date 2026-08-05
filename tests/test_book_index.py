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

    def _comments_in_file(self, page_name):
        with open(self.book.page(page_name).local_file_path('pcgts.json')) as f:
            return ((json.load(f).get('page', {}) or {}).get('comments', {}) or {}).get('comments', [])

    def test_comments_endpoints_match_the_files(self):
        expected = {p.page: len(self._comments_in_file(p.page)) for p in self.book.pages()}
        expected = {name: n for name, n in expected.items() if n > 0}
        self.assertTrue(expected, 'test storage must contain at least one commented page')

        response = self.client.get('/api/book/demo/comments/count', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertEqual(json.loads(response.content)['count'], sum(expected.values()))

        response = self.client.get('/api/book/demo/comments', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        data = json.loads(response.content)['data']
        # only commented pages are listed, and with their full payload
        self.assertEqual({d['page']: len(d['comments']['comments']) for d in data}, expected)

    def test_comments_index_self_heals_on_pcgts_change(self):
        page = self.book.page('page00000001')
        path = page.local_file_path('pcgts.json')
        with open(path) as f:
            original = f.read()
        before = json.loads(self.client.get('/api/book/demo/comments/count').content)['count']
        try:
            d = json.loads(original)
            d['page']['comments'] = {'comments': [{'id': 'c1', 'text': 'added', 'aabb': None}]}
            with open(path, 'w') as f:
                json.dump(d, f)

            count = json.loads(self.client.get('/api/book/demo/comments/count').content)['count']
            self.assertEqual(count, before + 1)
            data = json.loads(self.client.get('/api/book/demo/comments').content)['data']
            added = [c for c in data if c['page'] == 'page00000001']
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0]['comments']['comments'][0]['text'], 'added')
        finally:
            with open(path, 'w') as f:
                f.write(original)
            pcgts_cache.invalidate(path)
        self.assertEqual(json.loads(self.client.get('/api/book/demo/comments/count').content)['count'], before)

    def test_comments_backfilled_for_rows_predating_the_columns(self):
        book_index.index_book(self.book)
        expected = json.loads(self.client.get('/api/book/demo/comments/count').content)['count']
        # rows written before the comments columns existed carry null/0 at unchanged mtimes
        PageIndex.objects.filter(book__name='demo').update(comments=None, comments_count=0)
        self.assertEqual(book_index.book_comments_count(self.book), expected)
        self.assertFalse(PageIndex.objects.filter(book__name='demo', comments__isnull=True).exists())

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

    def test_rename_invalidates_pcgts_cache(self):
        page = self.book.page('page00000001')
        old_path = page.local_file_path('pcgts.json')
        pcgts_cache.clear()
        pcgts_cache.get(page)
        self.assertIn(old_path, pcgts_cache._cache)
        page.rename('page_cache_rename_tmp')
        try:
            self.assertNotIn(old_path, pcgts_cache._cache)
        finally:
            page.rename('page00000001')

    def test_delete_invalidates_pcgts_cache(self):
        import shutil
        page = self.book.page('page_cache_delete_tmp')
        shutil.copytree(self.book.page('page00000001').local_path(), page.local_path())
        try:
            pcgts_cache.clear()
            pcgts_cache.get(page)
            path = page.local_file_path('pcgts.json')
            self.assertIn(path, pcgts_cache._cache)
        finally:
            page.delete()
        self.assertNotIn(page.local_file_path('pcgts.json'), pcgts_cache._cache)

    def test_book_meta_write_is_atomic_and_clean(self):
        meta = self.book.get_meta()
        meta.to_file(self.book)
        leftovers = [f for f in os.listdir(self.book.local_path()) if f.endswith('.tmp')]
        self.assertEqual(leftovers, [])
        with open(self.book.local_path('book_meta.json')) as f:
            json.load(f)  # complete, valid JSON

    def test_stale_lock_auto_expires_on_access(self):
        from datetime import timedelta
        from django.utils import timezone
        page = self.book.page('page_test_lock')
        user = User.objects.get(username='user')
        page.lock(user)
        self.assertTrue(page.is_locked())
        PageEditLock.objects.filter(page__book__name='demo', page__name='page_test_lock') \
            .update(acquired_at=timezone.now() - timedelta(hours=13))
        self.assertFalse(page.is_locked())
        self.assertFalse(PageEditLock.objects.filter(page__book__name='demo', page__name='page_test_lock').exists())

    def test_release_stale_locks_command(self):
        from datetime import timedelta
        from django.core.management import call_command
        from django.utils import timezone
        user = User.objects.get(username='user')
        stale_page = self.book.page('page_test_lock')
        fresh_page = self.book.page('page00000001')
        stale_page.lock(user)
        fresh_page.lock(user)
        PageEditLock.objects.filter(page__name='page_test_lock') \
            .update(acquired_at=timezone.now() - timedelta(hours=13))
        try:
            call_command('release_stale_locks')
            self.assertFalse(PageEditLock.objects.filter(page__name='page_test_lock').exists())
            self.assertTrue(PageEditLock.objects.filter(page__name='page00000001').exists())
        finally:
            stale_page.release_lock()
            fresh_page.release_lock()
