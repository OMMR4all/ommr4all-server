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
    UNLOCKED = 'unlocked'


@dataclass
class PageSelectionParams(DataClassDictMixin):
    count: PageCount = PageCount.ALL
    pages: List[str] = field(default_factory=lambda: [])
    selected_pages_range_as_regex: str = ''


def check_if_page_range_regex_selector_valid(selected_pages: str):
    regex = r'(\s*\d+(\-\d+)?,)*(\s*\d+(\-\d+)?)'
    import re
    _rex = re.compile(regex)
    return True if _rex.fullmatch(selected_pages) else False


class PageSelection:
    def __init__(self,
                 book: DatabaseBook,
                 page_count: PageCount,
                 pages: Optional[List[DatabasePage]] = None,
                 pcgts: Optional[List[PcGts]] = None,
                 single_page: bool = False,
                 selected_pages_range_as_regex: str = ''
                 ):
        self.book = book
        self.page_count = page_count
        self.pages = pages if pages else []
        self.pcgts = pcgts
        self.single_page = single_page
        self.selected_pages_range_as_regex = selected_pages_range_as_regex
        self.total_pages = 0  # number of pages of the book, set by get_pages()

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
            [book.page(page) for page in params.pages],
            selected_pages_range_as_regex=params.selected_pages_range_as_regex
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

    def get_pages(self, unprocessed: Optional[Callable[[DatabasePage], bool]] = None,
                  unlocked: Optional[Callable[[DatabasePage], bool]] = None) -> List[DatabasePage]:
        if self.pcgts:
            self.total_pages = len(self.pcgts)
            return [DatabasePage(self.book, 'in_memory', skip_validation=True, pcgts=pcgts) for pcgts in self.pcgts]

        # One index-backed pass over the book instead of a page_progress.json parse per
        # page: both the `unlocked` predicate and the verified filter below read the
        # progress, and this endpoint runs on every render of the workflow tab.
        from database.book_index import prefill_page_progress
        book_pages = prefill_page_progress(self.book)
        self.total_pages = len(book_pages)
        by_name = {p.page: p for p in book_pages}

        def with_progress(pages: List[DatabasePage]) -> List[DatabasePage]:
            # Keep the caller's page object and only fill in the progress. The caller may
            # carry state the index cannot rebuild -- above all an in-memory pcgts posted
            # by the editor for a single-page operation (OperationView.op_to_task_runner).
            # Replacing the object here made every predictor re-read pcgts.json from disk,
            # so chaining layout -> symbols in the editor ran on the last saved state.
            for p in pages:
                if p.has_page_progress():
                    continue
                indexed = by_name.get(p.page)
                if indexed is not None:
                    p.set_page_progress(indexed.page_progress())
            return list(pages)

        def page_count_pages() -> List[DatabasePage]:
            if self.page_count == PageCount.ALL:
                return book_pages
            elif self.page_count == PageCount.UNPROCESSED:
                if unprocessed:
                    return [p for p in book_pages if unprocessed(p)]
                else:
                    return book_pages
            elif self.page_count == PageCount.UNLOCKED:
                if unlocked:
                    return [p for p in book_pages if unlocked(p)]
                else:
                    return book_pages
            elif self.page_count == PageCount.CUSTOM:
                if check_if_page_range_regex_selector_valid(self.selected_pages_range_as_regex):
                    pages = book_pages
                    selected_pages = []
                    selected = self.selected_pages_range_as_regex.replace(" ", "")
                    page_ranges = selected.split(",")
                    for i in page_ranges:
                        values = i.split("-")
                        if len(values) == 2:
                            r0 = int(values[0]) - 1
                            r1 = int(values[1])
                            if r0 >= 0:
                                if r0 <= r1:
                                    if r0 <= len(pages) and r1 <= len(pages):
                                        selected_pages += pages[r0:r1]
                        else:
                            r0 = int(values[0]) - 1
                            if 0 <= r0 < len(pages):
                                selected_pages += [pages[r0]]
                    return list(set(selected_pages))
                else:
                    return with_progress(self.pages)

            else:
                return with_progress(self.pages)

        if self.single_page:
            # An explicit single-page run comes from the editor, where re-running on an
            # already verified page is deliberate. Filtering it out here would surface as
            # "produced no result for page ..." (see TaskRunnerPrediction.run).
            return page_count_pages()

        return [page for page in page_count_pages() if not page.page_progress().verified]

    def get_pcgts(self, unprocessed: Optional[Callable[[DatabasePage], bool]] = None) -> List[PcGts]:
        if self.pcgts:
            return self.pcgts
        else:
            return [p.pcgts() for p in self.get_pages(unprocessed)]


if __name__ == "__main__":
    print(check_if_page_range_regex_selector_valid("1-12, 3-15, 5, 4-5, 13, 4444, 4-14"))
    pass
