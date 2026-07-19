import json
import os

from django.core.management.base import BaseCommand

from database.database_book import DatabaseBook
from database.book_index import index_book_meta, index_page, index_documents, _mtime
from database.models.book_index import BookIndex, PageIndex, PageEditLock


class Command(BaseCommand):
    help = 'Rebuild the DB index of the on-disk book storage. Idempotent: the storage ' \
           'folder stays the source of truth, rows are upserted keyed by file mtimes. ' \
           'Run after copying book folders from another server (a missing index also ' \
           'self-heals lazily on first access).'

    def add_arguments(self, parser):
        parser.add_argument('books', nargs='*', help='Book ids to reindex (default: all)')
        parser.add_argument('--full', action='store_true',
                            help='Rescan every page even if the file mtimes match the index')
        parser.add_argument('--no-prune', action='store_true',
                            help='Keep index rows of books/pages whose folders no longer exist')
        parser.add_argument('--import-locks', action='store_true',
                            help='Convert legacy .lock files of pages into DB edit locks and delete them')
        parser.add_argument('--warm-counts', action='store_true',
                            help='Also precompute the per-page symbol/line counts so the first '
                                 'stats request of a freshly indexed book is served warm')
        parser.add_argument('--dry-run', action='store_true',
                            help='Only print what would be done')

    def handle(self, *args, **options):
        books = DatabaseBook.list_available()
        if options['books']:
            books = [b for b in books if b.book in options['books']]
            missing = set(options['books']) - {b.book for b in books}
            for name in sorted(missing):
                self.stdout.write(self.style.WARNING('{}: no such book folder'.format(name)))

        for book in books:
            try:
                self._reindex_book(book, options)
            except Exception as e:
                self.stderr.write(self.style.ERROR('{}: reindex failed: {}'.format(book.book, e)))

        if not options['no_prune'] and not options['books'] and not options['dry_run']:
            stale = BookIndex.objects.exclude(name__in=[b.book for b in books])
            for row in stale:
                self.stdout.write('{}: folder gone, removing index rows'.format(row.name))
            stale.delete()

        self.stdout.write(self.style.SUCCESS('done'))

    def _reindex_book(self, book: DatabaseBook, options):
        if options['dry_run']:
            self.stdout.write('{}: would reindex {} page(s)'.format(book.book, len(book.pages())))
            return

        book_row = index_book_meta(book)
        seed = {} if options['full'] else self._load_stats_cache_seed(book)
        n_pages = 0
        page_names = []
        for page in book.pages():
            page_names.append(page.page)
            self._seed_page_row(book_row, page, seed)
            index_page(page, book_row=book_row, force=options['full'])
            n_pages += 1
        if not options['no_prune']:
            book_row.pages.exclude(name__in=page_names).delete()
        index_documents(book, book_row=book_row)

        if options['import_locks']:
            self._import_locks(book, book_row)

        if options['warm_counts']:
            from database.book_index import book_counts
            book_counts(book)  # computes and stores counts of every changed page

        self.stdout.write('{}: {} page(s) indexed'.format(book.book, n_pages))

    @staticmethod
    def _load_stats_cache_seed(book: DatabaseBook) -> dict:
        """Entries of the legacy book_overview_stats_cache.json: reusing them avoids
        re-reading every pcgts.json of a large storage during the first reindex."""
        try:
            with open(book.local_path('book_overview_stats_cache.json')) as f:
                cache = json.load(f)
            if cache.get('version') == 1 and isinstance(cache.get('pages'), dict):
                return cache['pages']
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    @staticmethod
    def _legacy_mtime(path: str) -> float:
        # legacy book_overview_stats_cache.json stored float-second mtimes
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    @classmethod
    def _seed_page_row(cls, book_row: BookIndex, page, seed: dict):
        """Pre-create the page row from a matching legacy cache entry so index_page's
        mtime check succeeds without opening pcgts.json."""
        entry = seed.get(page.page)
        if not entry:
            return
        if entry.get('pcgts_mtime') != cls._legacy_mtime(page.local_file_path('pcgts.json')) \
                or entry.get('progress_mtime') != cls._legacy_mtime(page.local_file_path('page_progress.json')):
            return
        pcgts_mtime = _mtime(page.local_file_path('pcgts.json'))
        progress_mtime = _mtime(page.local_file_path('page_progress.json'))
        if PageIndex.objects.filter(book=book_row, name=page.page).exists():
            return
        PageIndex.objects.create(
            book=book_row, name=page.page,
            pcgts_mtime=pcgts_mtime, progress_mtime=progress_mtime,
            has_symbols=bool(entry.get('has_symbols')),
            verified=bool(entry.get('verified')),
            progress_locks=entry.get('locked') or {},
        )

    def _import_locks(self, book: DatabaseBook, book_row: BookIndex):
        from django.contrib.auth.models import User
        for page in book.pages():
            lock_path = page.local_file_path('.lock')
            if not os.path.exists(lock_path):
                continue
            with open(lock_path) as f:
                username = f.read().strip()
            try:
                user = User.objects.get(username=username)
                page_row = PageIndex.objects.get(book=book_row, name=page.page)
                PageEditLock.objects.update_or_create(page=page_row, defaults={'user': user})
                self.stdout.write('{}/{}: imported edit lock of {}'.format(book.book, page.page, username))
            except User.DoesNotExist:
                self.stdout.write(self.style.WARNING(
                    '{}/{}: dropping lock of unknown user {}'.format(book.book, page.page, username)))
            os.remove(lock_path)
