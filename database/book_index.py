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
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

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


def _pcgts_comments(path: str) -> dict:
    """The raw `page.comments` payload of a pcgts file ({} if it carries none).

    Same trick as _pcgts_has_symbols: plain json, no geometry parsing."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    comments = (d.get('page', {}) or {}).get('comments', {}) or {}
    return comments if comments.get('comments') else {}


def _make_aware(dt) -> Optional['timezone.datetime']:
    if dt is None:
        return None
    if timezone.is_naive(dt):
        return timezone.make_aware(dt)
    return dt


def safe(func):
    """An index update is a cache write: retry once, then log and swallow the failure —
    the mtime validation on read repairs any dropped update.

    Losing a write race ('database is locked') is expected under concurrent savers and
    the retry almost always settles it, so it is reported as a single warning line;
    anything else keeps its traceback."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from django.db import DatabaseError
        from database.db_errors import retry_on_db_error
        try:
            return retry_on_db_error(lambda: func(*args, **kwargs))
        except DatabaseError as e:
            # expected under concurrency and self-healing, so one line, no traceback:
            # describe_db_error already names the extended code and the database state
            from database.db_errors import log_db_failure
            log_db_failure(logger, 'Book index update {} (will self-heal on read)'.format(
                func.__name__), e)
            return None
        except Exception as e:
            logger.warning('Book index update {} failed (will self-heal on read): {}'.format(
                func.__name__, e))
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
        pcgts_path = db_page.local_file_path('pcgts.json')
        defaults['has_symbols'] = pcgts_mtime > 0 and _pcgts_has_symbols(pcgts_path)
        comments = _pcgts_comments(pcgts_path) if pcgts_mtime > 0 else {}
        defaults['comments'] = comments
        defaults['comments_count'] = len(comments.get('comments', []))
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
    # e.is_dir() already covers the exists/isdir half of DatabaseBook.is_valid()
    with os.scandir(settings.PRIVATE_MEDIA_ROOT) as it:
        names = [e.name for e in it
                 if e.is_dir() and DatabaseBook(e.name, skip_validation=True).is_valid_name()]
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


PAGE_ROW_FIELDS = ('name', 'pcgts_mtime', 'progress_mtime', 'has_symbols', 'verified',
                   'progress_locks', 'counts', 'counts_mtime', 'comments', 'comments_count')


@dataclass(frozen=True)
class PageRow:
    """Read-only view of a PageIndex row, as handed out by sync_book_pages.

    Materialising real PageIndex models for a whole book cost 136 ms for 2657 pages
    where the same rows via .values() cost 16 ms — and the common all-fresh case only
    ever reads. Writers (book_counts, book_comments) persist with a queryset .update()
    instead of row.save()."""
    name: str
    pcgts_mtime: int
    progress_mtime: int
    has_symbols: bool
    verified: bool
    progress_locks: dict
    counts: Optional[dict]
    counts_mtime: int
    comments: Optional[dict]
    comments_count: int

    @staticmethod
    def from_model(row: PageIndex) -> 'PageRow':
        return PageRow(**{f: getattr(row, f) for f in PAGE_ROW_FIELDS})


def sync_book_pages(db_book: 'DatabaseBook', book_row: Optional[BookIndex] = None) -> List[PageRow]:
    """Stat-validated page rows of the book, in page order (pruned).

    All existing rows are fetched in one query; unchanged pages (the common case) cause
    no further DB access, so a synced book costs 1 query + 2 stats/page. Pass `book_row`
    when the caller already resolved it (the books listing does, for every book)."""
    if book_row is None:
        book_row = index_book_meta(db_book)
    existing = {d['name']: d for d in
                PageIndex.objects.filter(book=book_row).values(*PAGE_ROW_FIELDS)}
    rows = []
    page_names = []
    for name in db_book.page_names_on_disk():
        page_names.append(name)
        db_page = db_book.page(name)
        cached = existing.get(name)
        if cached is not None \
                and cached['pcgts_mtime'] == _mtime(db_page.local_file_path('pcgts.json')) \
                and cached['progress_mtime'] == _mtime(db_page.local_file_path('page_progress.json')):
            rows.append(PageRow(**cached))
        else:
            # new or changed: index_page rescans the changed files and writes the row
            rows.append(PageRow.from_model(index_page(db_page, book_row=book_row)))
    stale = set(existing) - set(page_names)
    if stale:
        book_row.pages.filter(name__in=stale).delete()
    return rows


def stored_book_pages(db_book: 'DatabaseBook') -> List[PageRow]:
    """Page rows exactly as stored, without stat-validating or writing anything.

    For read paths that only *display* progress: sync_book_pages stats every page and
    upserts the changed ones, which turns a plain page view into a writer -- multiplied by
    every open browser tab. The stored values are refreshed by the next real sync (a page
    save indexes its own page), so displaying them costs one query and no writes."""
    return [PageRow(**d) for d in PageIndex.objects
            .filter(book__name=db_book.book).order_by('name').values(*PAGE_ROW_FIELDS)]


def _store_page_fields(db_book: 'DatabaseBook', name: str, **fields):
    """Best-effort write-back of a lazily computed field (see PageRow)."""
    try:
        PageIndex.objects.filter(book__name=db_book.book, name=name).update(**fields)
    except Exception as e:
        logger.warning('Could not store {} of {}/{}'.format(
            ', '.join(fields), db_book.book, name))
        logger.exception(e)


def pages_with_lock(db_book: 'DatabaseBook', locks) -> list:
    """DatabasePages whose progress locks match all requested LockStates.

    Index-backed replacement of the page_progress.json loop: only pages whose
    files changed since the last sync are reparsed."""
    from database.file_formats.performance.pageprogress import Locks
    rows = sync_book_pages(db_book)
    wanted = [(Locks(l.label).value, l.lock) for l in locks]
    return [db_book.page(row.name) for row in rows
            if all(bool((row.progress_locks or {}).get(label, False)) == lock for label, lock in wanted)]


def prefill_page_progress(db_book: 'DatabaseBook') -> list:
    """DatabasePages of the book with their page_progress preloaded from the index.

    Callers can then run the progress-based predicates (`verified`, `unlocked`) over a
    whole book without opening — or, via create_if_not_existing, writing — a single
    page_progress.json. The index rows are stat-validated, so the values are the same
    ones PageProgress.from_json_file would return."""
    from database.file_formats.performance.pageprogress import Locks, PageProgress
    pages = []
    for row in sync_book_pages(db_book):
        page = db_book.page(row.name)
        locks = row.progress_locks or {}
        page.set_page_progress(PageProgress(
            locked={l: bool(locks.get(l.value, False)) for l in Locks},
            verified=row.verified,
        ))
        pages.append(page)
    return pages


def book_counts(db_book: 'DatabaseBook'):
    """Symbol/line/text counts of the whole book, computed per page and cached in
    the index: only pages whose pcgts changed since the last call are reparsed."""
    import dataclasses
    from database.tools.book_statistics import Counts, count_page

    total = Counts()
    for row in sync_book_pages(db_book):
        if row.counts is None or row.counts_mtime != row.pcgts_mtime:
            counts = count_page(db_book.page(row.name).pcgts_cached())
            _store_page_fields(db_book, row.name,
                               counts=counts.to_dict(), counts_mtime=row.pcgts_mtime)
        else:
            counts = Counts.from_dict(row.counts)
        for f in dataclasses.fields(Counts):
            setattr(total, f.name, getattr(total, f.name) + getattr(counts, f.name))
    return total


def book_comments(db_book: 'DatabaseBook') -> list:
    """(page_name, comments dict) of every page of the book that carries comments.

    Index-backed replacement of the full-book PcGts parse: a synced book costs
    1 query + 2 stats/page. Rows predating the comments columns have
    `comments is None` and are backfilled here (once) instead of requiring a reindex."""
    out = []
    for row in sync_book_pages(db_book):
        comments, count = row.comments, row.comments_count
        if comments is None:
            comments = _pcgts_comments(db_book.page(row.name).local_file_path('pcgts.json')) \
                if row.pcgts_mtime > 0 else {}
            count = len(comments.get('comments', []))
            _store_page_fields(db_book, row.name, comments=comments, comments_count=count)
        if count > 0:
            out.append((row.name, comments))
    return out


def book_comments_count(db_book: 'DatabaseBook') -> int:
    return sum(len(comments.get('comments', [])) for _, comments in book_comments(db_book))


def get_documents_json(db_book: 'DatabaseBook') -> Optional[dict]:
    """The book_documents.json payload from the index, or None if it cannot be
    proven fresh (caller then falls back to update_book_documents_cached)."""
    # `documents` is the whole book_documents.json (megabytes on a large book), so prove
    # the row current from file_mtime alone before pulling that column out of the DB
    file_mtime = _mtime(db_book.local_path('book_documents.json'))
    if file_mtime == 0:
        return None
    if not BookDocumentsIndex.objects.filter(book__name=db_book.book, file_mtime=file_mtime)\
            .exclude(documents=None).exists():
        return None

    pages_dir = db_book.local_path('pages')
    page_names = db_book.page_names_on_disk()
    # fragment mtimes are part of the book_documents.json file format and are written as
    # float seconds (os.path.getmtime) — compare in the same unit
    page_mtimes = {}
    for name in page_names:
        try:
            page_mtimes[name] = os.stat(os.path.join(pages_dir, name, 'pcgts.json')).st_mtime
        except OSError:
            return None  # a page without pcgts cannot match a fragment

    documents = BookDocumentsIndex.objects.filter(book__name=db_book.book)\
        .values_list('documents', flat=True).first()
    if documents is None \
            or documents.get('database_documents') is None or documents.get('page_fragments') is None:
        return None

    # function-local import: database_book_documents imports this module
    from database.database_book_documents import DatabaseBookDocuments
    if documents.get('version', 0) != DatabaseBookDocuments.DOCUMENTS_FORMAT_VERSION:
        # assembled by an older version of the code — let the caller reassemble once
        return None

    fragment_mtimes = {f.get('page_name'): f.get('mtime') for f in documents['page_fragments']}
    if len(page_names) != len(fragment_mtimes):
        return None
    if any(fragment_mtimes.get(name) != mtime for name, mtime in page_mtimes.items()):
        return None

    return documents


safe_index_book_meta = safe(index_book_meta)
safe_index_page = safe(index_page)
safe_index_book = safe(index_book)
safe_index_documents = safe(index_documents)
safe_remove_book = safe(remove_book)
safe_remove_page = safe(remove_page)
safe_ensure_book_indexed = safe(ensure_book_indexed)
