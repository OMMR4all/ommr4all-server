from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json, LetterCase

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import SymbolType, BlockType
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
    n_chars: int = 0
    n_syllabels: int = 0
    n_words: int = 0
    n_drop_capitals: int = 0

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
            if  len([s for s in ml.symbols if s.symbol_type == SymbolType.ACCID]) > 0:
                print(pcgts.page.location.local_path())
                for s in ml.symbols:
                    if s.symbol_type == SymbolType.ACCID:
                        print(s.accid_type)
        tls = p.all_text_lines()
        for tl in tls:
            counts.n_syllabels += len(tl.sentence.syllables)
            counts.n_chars += len(tl.text())
            counts.n_words += len(tl.text().split(" "))

        counts.n_drop_capitals += len(p.blocks_of_type(BlockType.DROP_CAPITAL))

        if callback:
            callback.updated(counts, i + 1, len(pages))

    return counts
