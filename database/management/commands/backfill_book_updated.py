import os
from datetime import datetime

from django.core.management.base import BaseCommand

from database.database_book import DatabaseBook

# the files a user edit actually touches; derived images etc. are ignored
CONTENT_FILES = ['pcgts.json', 'page_progress.json']


def latest_content_mtime(book: DatabaseBook) -> float:
    latest = 0
    for page in book.pages():
        for f in CONTENT_FILES:
            path = page.local_file_path(f)
            try:
                latest = max(latest, os.path.getmtime(path))
            except OSError:
                pass
        if latest == 0:
            # page without annotations yet: fall back to its directory (upload time)
            try:
                latest = max(latest, os.path.getmtime(page.local_path()))
            except OSError:
                pass

    if latest == 0:
        try:
            latest = os.path.getmtime(book.local_path())
        except OSError:
            pass

    return latest


class Command(BaseCommand):
    help = 'Backfill the last-modified timestamp (book_meta.json "updated") of books that do not have one yet, ' \
           'derived from the modification times of their page annotation files.'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true',
                            help='Recompute the timestamp even for books that already have one')
        parser.add_argument('--dry-run', action='store_true',
                            help='Only print what would be written')

    def handle(self, *args, **options):
        n_set = n_skipped = 0
        for book in DatabaseBook.list_available():
            meta = book.get_meta()
            if meta.updated is not None and not options['force']:
                n_skipped += 1
                continue

            mtime = latest_content_mtime(book)
            if mtime == 0:
                self.stdout.write(self.style.WARNING('{}: no timestamp derivable, skipping'.format(book.book)))
                n_skipped += 1
                continue

            updated = datetime.fromtimestamp(mtime)
            self.stdout.write('{}: updated = {}'.format(book.book, updated.isoformat()))
            if not options['dry_run']:
                meta.updated = updated
                meta.to_file(book)
            n_set += 1

        self.stdout.write(self.style.SUCCESS('{} book(s) backfilled, {} skipped{}'.format(
            n_set, n_skipped, ' (dry run, nothing written)' if options['dry_run'] else '')))
