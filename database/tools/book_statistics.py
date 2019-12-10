from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json, LetterCase

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import SymbolType
from abc import ABC, abstractmethod


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Counts:
    n_pages: int = 0
    n_staves: int = 0
    n_staff_lines: int = 0
    n_symbols: int = 0
    n_note_components: int = 0
    n_clefs: int = 0
    n_accids: int = 0


class Callback(ABC):
    @abstractmethod
    def updated(self, counts: Counts, n_processed, n_total):
        pass


def compute_books_statistics(books: List[DatabaseBook], ignore_page: List[str] = None, callback: Callback = None) -> Counts:
    ignore_page = ignore_page if ignore_page else []
    pages = sum([[page.pcgts() for page in book.pages() if not any([s in page.page for s in ignore_page])] for book in books], [])
    return get_counts(pages, callback)


def compute_book_statistics(book: DatabaseBook, ignore_page: List[str] = None, callback: Callback = None) -> Counts:
    return compute_books_statistics([book], ignore_page, callback)


def get_counts(pages: List[PcGts], callback: Callback = None) -> Counts:
    counts = Counts()

    if callback:
        callback.updated(counts, 0, len(pages))

    for i, pcgts in enumerate(pages):
        p = pcgts.page
        counts.n_pages += 1
        mls = p.all_music_lines()
        counts.n_staves += len(mls)
        for ml in mls:
            counts.n_staff_lines += len(ml.staff_lines)
            counts.n_symbols += len(ml.symbols)
            counts.n_note_components += len([s for s in ml.symbols if s.symbol_type == SymbolType.NOTE])
            counts.n_clefs += len([s for s in ml.symbols if s.symbol_type == SymbolType.CLEF])
            counts.n_accids += len([s for s in ml.symbols if s.symbol_type == SymbolType.ACCID])

        if callback:
            callback.updated(counts, i + 1, len(pages))

    return counts
