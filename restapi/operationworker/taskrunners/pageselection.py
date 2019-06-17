from database.database_page import DatabaseBook, DatabasePage
from typing import Optional, List, Tuple, Callable
from enum import Enum
from restapi.utils.jsonparsing import JsonParseKeyNotFound, require_json


class PageCount(Enum):
    ALL = 'all'
    UNPROCESSED = 'unprocessed'
    CUSTOM = 'custom'


class PageSelection:
    def __init__(self,
                 book: DatabaseBook,
                 page_count: PageCount,
                 pages: Optional[List[DatabasePage]] = None,
                 single_page: bool = False,
                 ):
        self.book = book
        self.page_count = page_count
        self.pages = pages if pages else []
        self.single_page = single_page

    @staticmethod
    def from_page(page: DatabasePage):
        return PageSelection(
            page.book,
            PageCount.CUSTOM,
            [page],
            single_page=True
        )

    @staticmethod
    def from_json(d: dict, book: DatabaseBook):
        if not 'count' in d:
            raise JsonParseKeyNotFound('count', d)

        return PageSelection(
            book,
            PageCount(require_json(d, 'count')),
            [book.page(page) for page in d.get('pages', [])]
        )

    def identifier(self) -> Tuple:
        return self.book, self.page_count, self.pages

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.identifier() == other.identifier()

    def get(self, unprocessed: Optional[Callable[[DatabasePage], bool]] = None) -> List[DatabasePage]:
        if self.page_count == PageCount.ALL:
            return self.book.pages()
        elif self.page_count == PageCount.UNPROCESSED:
            if unprocessed:
                return [p for p in self.book.pages() if unprocessed(p)]
            else:
                return self.book.pages()
        else:
            return self.pages

