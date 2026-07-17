import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING

from database.file_formats.performance.pageprogress import Locks, PageProgress

if TYPE_CHECKING:
    from database.database_book import DatabaseBook
    from database.database_page import DatabasePage

logger = logging.getLogger(__name__)

CACHE_FILE = 'book_overview_stats_cache.json'
CACHE_VERSION = 1

STATE_EMPTY = 'empty'
STATE_NO_TRANSCRIPTION = 'no_transcription'
STATE_TRANSCRIPTION_UNCORRECTED = 'transcription_uncorrected'
STATE_PARTIALLY_CORRECTED = 'partially_corrected'
STATE_FULLY_CORRECTED = 'fully_corrected'


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
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


def _scan_page(page: 'DatabasePage', pcgts_mtime: float, progress_mtime: float) -> dict:
    entry = {
        'pcgts_mtime': pcgts_mtime,
        'progress_mtime': progress_mtime,
        'has_symbols': False,
        'locked': {l.value: False for l in Locks},
        'verified': False,
    }

    if pcgts_mtime > 0:
        entry['has_symbols'] = _pcgts_has_symbols(page.local_file_path('pcgts.json'))

    if progress_mtime > 0:
        pp = PageProgress.from_json_file(page.local_file_path('page_progress.json'))
        entry['locked'] = {l.value: bool(pp.locked.get(l, False)) for l in Locks}
        entry['verified'] = pp.verified

    return entry


def _load_cache(path: str) -> dict:
    try:
        with open(path) as f:
            cache = json.load(f)
        if cache.get('version') == CACHE_VERSION and isinstance(cache.get('pages'), dict):
            return cache['pages']
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _store_cache(book: 'DatabaseBook', path: str, pages: dict):
    # best effort: the stats are recomputable, so never fail the request on a cache write
    try:
        fd, tmp_path = tempfile.mkstemp(dir=book.local_path(), suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump({'version': CACHE_VERSION, 'pages': pages}, f)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning('Could not write overview stats cache of book {}'.format(book.book))
        logger.exception(e)


def derive_state(n_pages: int, n_pages_with_symbols: int, n_total_locks: int) -> str:
    if n_pages == 0:
        return STATE_EMPTY
    if n_total_locks == 0:
        if n_pages_with_symbols == 0:
            return STATE_NO_TRANSCRIPTION
        return STATE_TRANSCRIPTION_UNCORRECTED
    if n_total_locks == len(Locks) * n_pages:
        return STATE_FULLY_CORRECTED
    return STATE_PARTIALLY_CORRECTED


def compute_overview_stats(book: 'DatabaseBook') -> dict:
    pages = book.pages()
    cache_path = book.local_path(CACHE_FILE)
    cached_pages = _load_cache(cache_path)

    entries = {}
    changed = False
    for page in pages:
        pcgts_mtime = _mtime(page.local_file_path('pcgts.json'))
        progress_mtime = _mtime(page.local_file_path('page_progress.json'))

        entry = cached_pages.get(page.page)
        if entry is None or entry.get('pcgts_mtime') != pcgts_mtime or entry.get('progress_mtime') != progress_mtime:
            entry = _scan_page(page, pcgts_mtime, progress_mtime)
            changed = True

        entries[page.page] = entry

    # detect removed pages
    changed = changed or len(entries) != len(cached_pages)

    if changed:
        _store_cache(book, cache_path, entries)

    n_pages = len(entries)
    lock_counts = {l.value: 0 for l in Locks}
    n_pages_with_symbols = 0
    n_verified = 0
    for entry in entries.values():
        if entry.get('has_symbols'):
            n_pages_with_symbols += 1
        if entry.get('verified'):
            n_verified += 1
        locked = entry.get('locked', {})
        for l in Locks:
            if locked.get(l.value):
                lock_counts[l.value] += 1

    n_total_locks = sum(lock_counts.values())

    return {
        'pages': n_pages,
        'pagesWithSymbols': n_pages_with_symbols,
        'verified': n_verified,
        'locks': lock_counts,
        'percentages': {k: round(v * 100 / n_pages, 1) if n_pages > 0 else 0 for k, v in lock_counts.items()},
        'state': derive_state(n_pages, n_pages_with_symbols, n_total_locks),
    }
