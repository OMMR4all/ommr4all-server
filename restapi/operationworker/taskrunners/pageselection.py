from database.database_page import DatabaseBook, DatabasePage
from database.file_formats.pcgts import PcGts
from typing import Optional, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from mashumaro import DataClassDictMixin
from shared.jsonparsing import JsonParseKeyNotFound, require_json


class PageCount(Enum):
    ALL = 'all'
    UNPROCESSED = 'unprocessed'
    CUSTOM = 'custom'


@dataclass
class PageSelectionParams(DataClassDictMixin):
    count: PageCount = PageCount.ALL
    pages: List[str] = field(default_factory=lambda: [])


class PageSelection:
    def __init__(self,
                 book: DatabaseBook,
                 page_count: PageCount,
                 pages: Optional[List[DatabasePage]] = None,
                 pcgts: Optional[List[PcGts]] = None,
                 single_page: bool = False,
                 ):
        self.book = book
        self.page_count = page_count
        self.pages = pages if pages else []
        self.pcgts = pcgts
        self.single_page = single_page

        if pcgts:
            self.pages = [p.page.location for p in pcgts]

    @staticmethod
    def from_book(book: DatabaseBook):
        return PageSelection(book, PageCount.ALL)

    @staticmethod
    def from_params(params: PageSelectionParams, book: DatabaseBook):
        return PageSelection(
            book,
            PageCount(params.count),
            [book.page(page) for page in params.pages]
        )

    @staticmethod
    def from_page(page: DatabasePage):
        return PageSelection(
            page.book,
            PageCount.CUSTOM,
            [page],
            single_page=True
        )

    @staticmethod
    def from_pcgts(pcgts: PcGts):
        return PageSelection(
            pcgts.page.location.book,
            PageCount.CUSTOM,
            pcgts=[pcgts],
            single_page=True,
        )

    @staticmethod
    def from_dict(d: dict, book: DatabaseBook):
        return PageSelection(
            book,
            PageCount(d.get('count', PageCount.ALL.value)),
            [book.page(page) for page in d.get('pages', [])]
        )

    def identifier(self) -> Tuple:
        return self.book, self.page_count, self.pages

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.identifier() == other.identifier()

    def get_pages(self, unprocessed: Optional[Callable[[DatabasePage], bool]] = None) -> List[DatabasePage]:
        if self.pcgts:
            return [DatabasePage(self.book, 'in_memory', skip_validation=True, pcgts=pcgts) for pcgts in self.pcgts]

        def page_count_pages() -> List[DatabasePage]:
            if self.page_count == PageCount.ALL:
                return self.book.pages()
            elif self.page_count == PageCount.UNPROCESSED:
                if unprocessed:
                    return [p for p in self.book.pages() if unprocessed(p)]
                else:
                    return self.book.pages()
            else:
                return self.pages

        return [page for page in page_count_pages() if not page.page_progress().verified]

    def get_pcgts(self, unprocessed: Optional[Callable[[DatabasePage], bool]] = None) -> List[PcGts]:
        if self.pcgts:
            return self.pcgts
        else:
            return [p.pcgts() for p in self.get_pages(unprocessed)]
