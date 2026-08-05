from typing import TYPE_CHECKING

from database.file_formats.performance.pageprogress import Locks

if TYPE_CHECKING:
    from database.database_book import DatabaseBook

STATE_EMPTY = 'empty'
STATE_NO_TRANSCRIPTION = 'no_transcription'
STATE_TRANSCRIPTION_UNCORRECTED = 'transcription_uncorrected'
STATE_PARTIALLY_CORRECTED = 'partially_corrected'
STATE_FULLY_CORRECTED = 'fully_corrected'


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


def compute_overview_stats(book: 'DatabaseBook', book_row=None) -> dict:
    # the page index revalidates each page by mtime and reparses only changed ones
    from database.book_index import sync_book_pages
    rows = sync_book_pages(book, book_row=book_row)

    n_pages = len(rows)
    lock_counts = {l.value: 0 for l in Locks}
    n_pages_with_symbols = 0
    n_verified = 0
    for row in rows:
        if row.has_symbols:
            n_pages_with_symbols += 1
        if row.verified:
            n_verified += 1
        locked = row.progress_locks or {}
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
