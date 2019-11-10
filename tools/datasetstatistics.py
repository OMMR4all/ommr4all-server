from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List
from prettytable import PrettyTable
from mashumaro import DataClassDictMixin

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import SymbolType


def list_all_pcgts(books: List[str], ignore_page: List[str]):
    all_stats = []
    for book in books:
        book = DatabaseBook(book)
        for page in book.pages():
            if any([s in page.page for s in ignore_page]):
                continue
            all_stats.append(page.pcgts())

    return all_stats


@dataclass
class Counts(DataClassDictMixin):
    n_pages: int = 0
    n_staves: int = 0
    n_staff_lines: int = 0
    n_symbols: int = 0
    n_note_components: int = 0
    n_clefs: int = 0
    n_accids: int = 0


def get_counts(pages: List[PcGts]):
    counts = Counts()

    for pcgts in pages:
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

    return counts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--ignore-page", nargs="+", default=[])

    args = parser.parse_args()

    all_book_counts = [get_counts(list_all_pcgts([book], args.ignore_page)) for book in args.books]

    pt = PrettyTable([n for n, _ in all_book_counts[0].to_dict().items()])
    for book_counts in all_book_counts:
        pt.add_row([v for _, v in book_counts.to_dict().items()])

    print(pt)


