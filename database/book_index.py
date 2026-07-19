"""Synchronisation between the on-disk book storage and its queryable DB mirror.

The storage folder stays the source of truth (a book folder is self-contained and
portable between servers); the BookIndex/PageIndex/BookDocumentsIndex rows are pure
derived data keyed by file mtimes. Rows are written best-effort — an index update
must never fail the actual file write — and reads validate mtimes, so a stale or
missing index self-heals on access or via `manage.py reindex_books`.
"""
import functools
import json
import logging
import os
from typing import TYPE_CHECKING, Optional

from django.utils import timezone

from database.models.book_index import BookIndex, PageIndex, BookDocumentsIndex

if TYPE_CHECKING:
    from database.database_book import DatabaseBook
    from database.database_page import DatabasePage

logger = logging.getLogger(__name__)


# sentinel for "caller already looked the row up" (None means "no row exists")
_UNSET = object()


def _mtime(path: str) -> int:
    """File mtime in integer nanoseconds (0 = file absent).

    Nanosecond resolution matches pcgts_cache: two writes within the same second
    must not leave the index stale (float seconds would miss the second one)."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0


def _pcgts_has_symbols(path: str) -> bool:
    # raw json access instead of PcGts.from_file: no geometry parsing required
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    for block in d.get('page', {}).get('blocks', []) or []:
        for line in block.get('lines', []) or []:
            if line.get('symbols'):
                return True
    return False


def _make_aware(dt) -> Optional['timezone.datetime']:
    if dt is None:
        return None
    if timezone.is_naive(dt):
        return timezone.make_aware(dt)
    return dt


def safe(func):
    """An index update is a cache write: log and swallow every failure
    (including 'database is locked' from concurrent SQLite writers) —
    the mtime validation on read repairs any dropped update."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning('Book index update {} failed (will self-heal on read)'.format(func.__name__))
            logger.exception(e)
            return None
    return wrapper


def index_book_meta(db_book: 'DatabaseBook') -> BookIndex:
    """Upsert the BookIndex row from book_meta.json (pages untouched).

    The stored `meta` is the normalized DatabaseBookMeta.to_dict() (defaults filled
    in), so serving it is byte-identical to loading and serializing the file."""
    meta_path = db_book.local_path('book_meta.json')
    meta_mtime = _mtime(meta_path)

    row = BookIndex.objects.filter(name=db_book.book).first()
    if row is not None and row.meta_mtime == meta_mtime:
        return row

    from database.database_book_meta import DatabaseBookMeta
    meta = DatabaseBookMeta.load(db_book)

    row, _ = BookIndex.objects.update_or_create(name=db_book.book, defaults={
        'meta': meta.to_dict(),
        'meta_mtime': meta_mtime,
        'display_name': meta.name or db_book.book,
        'notation_style': meta.notationStyle or '',
        'created': _make_aware(meta.created),
        'updated': _make_aware(meta.updated),
        'updated_by': meta.updatedBy or '',
    })
    return row


def index_page(db_page: 'DatabasePage', book_row: Optional[BookIndex] = None, force=False, row=_UNSET) -> PageIndex:
    """Upsert the PageIndex row; content is only rescanned when an mtime changed.

    Pass `row` (a PageIndex or None) when the caller already fetched the book's
    rows in bulk — this skips the per-page SELECT entirely."""
    if book_row is None:
        book_row = index_book_meta(db_page.book)

    pcgts_mtime = _mtime(db_page.local_file_path('pcgts.json'))
    progress_mtime = _mtime(db_page.local_file_path('page_progress.json'))

    if row is _UNSET:
        row = PageIndex.objects.filter(book=book_row, name=db_page.page).first()
    if row is not None and not force \
            and row.pcgts_mtime == pcgts_mtime and row.progress_mtime == progress_mtime:
        return row

    from database.file_formats.performance.pageprogress import Locks, PageProgress

    defaults = {'pcgts_mtime': pcgts_mtime, 'progress_mtime': progress_mtime}

    if row is None or force or row.pcgts_mtime != pcgts_mtime:
        defaults['has_symbols'] = pcgts_mtime > 0 and _pcgts_has_symbols(db_page.local_file_path('pcgts.json'))
        defaults['counts'] = None
        defaults['counts_mtime'] = 0

    if row is None or force or row.progress_mtime != progress_mtime:
        locked = {l.value: False for l in Locks}
        verified = False
        if progress_mtime > 0:
            pp = PageProgress.from_json_file(db_page.local_file_path('page_progress.json'))
            locked = {l.value: bool(pp.locked.get(l, False)) for l in Locks}
            verified = pp.verified
        defaults['progress_locks'] = locked
        defaults['verified'] = verified

    row, _ = PageIndex.objects.update_or_create(book=book_row, name=db_page.page, defaults=defaults)
    return row


def index_book(db_book: 'DatabaseBook', prune=True, force=False) -> BookIndex:
    """Upsert the book row and all of its page rows; prune rows of vanished pages."""
    book_row = index_book_meta(db_book)
    existing = {row.name: row for row in book_row.pages.all()}
    page_names = []
    for db_page in db_book.pages():
        page_names.append(db_page.page)
        index_page(db_page, book_row=book_row, force=force, row=existing.get(db_page.page))
    if prune:
        stale = set(existing) - set(page_names)
        if stale:
            book_row.pages.filter(name__in=stale).delete()
    return book_row


def index_documents(db_book: 'DatabaseBook', book_row: Optional[BookIndex] = None) -> Optional[BookDocumentsIndex]:
    """Mirror book_documents.json into its index row (stat-guarded)."""
    path = db_book.local_path('book_documents.json')
    file_mtime = _mtime(path)
    if file_mtime == 0:
        return None

    if book_row is None:
        book_row = index_book_meta(db_book)

    row = BookDocumentsIndex.objects.filter(book=book_row).first()
    if row is not None and row.file_mtime == file_mtime:
        return row

    try:
        with open(path) as f:
            documents = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    row, _ = BookDocumentsIndex.objects.update_or_create(book=book_row, defaults={
        'documents': documents,
        'file_mtime': file_mtime,
    })
    return row


def remove_book(book_name: str):
    BookIndex.objects.filter(name=book_name).delete()


def remove_page(book_name: str, page_name: str):
    PageIndex.objects.filter(book__name=book_name, name=page_name).delete()


def ensure_book_indexed(db_book: 'DatabaseBook') -> Optional[BookIndex]:
    """Lazy backfill: a freshly copied book folder becomes queryable on first access."""
    row = BookIndex.objects.filter(name=db_book.book).first()
    if row is None:
        row = index_book(db_book)
    return row


# ---------------------------------------------------------------------------
# Read paths: always stat-validated against the storage, so a stale or missing
# index (e.g. a book folder copied in from another server) self-heals inline.
# ---------------------------------------------------------------------------

def list_books_synced() -> list:
    """BookIndex rows of all valid book folders, meta-refreshed and pruned."""
    import ommr4all.settings as settings
    from database.database_book import DatabaseBook
    names = [name for name in os.listdir(settings.PRIVATE_MEDIA_ROOT)
             if DatabaseBook(name, skip_validation=True).is_valid()]
    if names:
        # hygiene only — rows absent from `names` are never returned anyway, and an
        # empty listing (e.g. storage mount hiccup) must not wipe the whole index
        BookIndex.objects.exclude(name__in=names).delete()
    return [index_book_meta(DatabaseBook(name)) for name in names]


def list_books_for_user(user, flag) -> list:
    """(meta dict, permission flags) of every book the user may access with `flag`."""
    from database.database_book import DatabaseBook
    out = []
    for row in list_books_synced():
        db_book = DatabaseBook(row.name)
        permissions = db_book.resolve_user_permissions(user)
        if permissions.has(flag):
            out.append((row.meta, permissions.flags))
    return out


def sync_book_pages(db_book: 'DatabaseBook') -> list:
    """Stat-validated PageIndex rows of the book, in page order (pruned).

    All existing rows are fetched in one query; unchanged pages (the common case)
    cause no further DB access, so a synced book costs 1 query + 2 stats/page."""
    book_row = index_book_meta(db_book)
    existing = {row.name: row for row in book_row.pages.all()}
    rows = []
    page_names = []
    for db_page in db_book.pages():
        page_names.append(db_page.page)
        rows.append(index_page(db_page, book_row=book_row, row=existing.get(db_page.page)))
    stale = set(existing) - set(page_names)
    if stale:
        book_row.pages.filter(name__in=stale).delete()
    return rows


def pages_with_lock(db_book: 'DatabaseBook', locks) -> list:
    """DatabasePages whose progress locks match all requested LockStates.

    Index-backed replacement of the page_progress.json loop: only pages whose
    files changed since the last sync are reparsed."""
    from database.file_formats.performance.pageprogress import Locks
    rows = sync_book_pages(db_book)
    wanted = [(Locks(l.label).value, l.lock) for l in locks]
    return [db_book.page(row.name) for row in rows
            if all(bool((row.progress_locks or {}).get(label, False)) == lock for label, lock in wanted)]


def book_counts(db_book: 'DatabaseBook'):
    """Symbol/line/text counts of the whole book, computed per page and cached in
    the index: only pages whose pcgts changed since the last call are reparsed."""
    import dataclasses
    from database.tools.book_statistics import Counts, count_page

    total = Counts()
    for row in sync_book_pages(db_book):
        if row.counts is None or row.counts_mtime != row.pcgts_mtime:
            counts = count_page(db_book.page(row.name).pcgts_cached())
            row.counts = counts.to_dict()
            row.counts_mtime = row.pcgts_mtime
            try:
                row.save(update_fields=['counts', 'counts_mtime'])
            except Exception as e:
                logger.warning('Could not store page counts of {}/{}'.format(db_book.book, row.name))
                logger.exception(e)
        else:
            counts = Counts.from_dict(row.counts)
        for f in dataclasses.fields(Counts):
            setattr(total, f.name, getattr(total, f.name) + getattr(counts, f.name))
    return total


def get_documents_json(db_book: 'DatabaseBook') -> Optional[dict]:
    """The book_documents.json payload from the index, or None if it cannot be
    proven fresh (caller then falls back to update_book_documents_cached)."""
    row = BookDocumentsIndex.objects.filter(book__name=db_book.book).first()
    if row is None or row.documents is None:
        return None
    if row.file_mtime != _mtime(db_book.local_path('book_documents.json')):
        return None

    documents = row.documents
    if documents.get('database_documents') is None or documents.get('page_fragments') is None:
        return None

    fragment_mtimes = {f.get('page_name'): f.get('mtime') for f in documents['page_fragments']}
    page_names = []
    for db_page in db_book.pages():
        page_names.append(db_page.page)
        # fragment mtimes are part of the book_documents.json file format and are
        # written as float seconds (os.path.getmtime) — compare in the same unit
        try:
            mtime = os.path.getmtime(db_page.local_file_path('pcgts.json'))
        except OSError:
            mtime = 0
        if mtime == 0 or fragment_mtimes.get(db_page.page) != mtime:
            return None
    if len(page_names) != len(fragment_mtimes):
        return None

    return documents


safe_index_book_meta = safe(index_book_meta)
safe_index_page = safe(index_page)
safe_index_book = safe(index_book)
safe_index_documents = safe(index_documents)
safe_remove_book = safe(remove_book)
safe_remove_page = safe(remove_page)
safe_ensure_book_indexed = safe(ensure_book_indexed)
