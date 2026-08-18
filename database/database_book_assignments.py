import functools
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, TYPE_CHECKING

from filelock import FileLock
from mashumaro import field_options
from mashumaro.mixins.json import DataClassJSONMixin

from database.database_book_meta import FormattedDateTime

if TYPE_CHECKING:
    from database.database_book import DatabaseBook

logger = logging.getLogger(__name__)

_assignments_file = 'page_assignments.json'

MAX_NOTE_LENGTH = 2000


@dataclass
class PageAssignment(DataClassJSONMixin):
    """A set of pages of a book that one user is responsible for.

    Pages are stored as explicit labels, not as a (start, end) range: page labels are
    arbitrary and the book order is 'sorted folder names', so a stored range would
    silently change its membership whenever pages are uploaded, renamed or deleted.
    Ranges are purely a presentation and input concern of the client.

    The assignee is stored by username (not as a foreign key) exactly like the per-book
    permissions in .permissions.pkl, so an exported book keeps its assignments.
    """
    id: str = ''
    username: str = ''
    pages: List[str] = field(default_factory=list)
    note: str = ''
    created: datetime = field(default_factory=lambda: datetime.now(),
                              metadata=field_options(serialization_strategy=FormattedDateTime()))
    createdBy: str = ''
    updated: Optional[datetime] = field(default=None,
                                        metadata=field_options(serialization_strategy=FormattedDateTime()))
    updatedBy: Optional[str] = None


@dataclass
class BookAssignments(DataClassJSONMixin):
    assignments: List[PageAssignment] = field(default_factory=list)

    def by_id(self, id: str) -> Optional[PageAssignment]:
        return next((a for a in self.assignments if a.id == id), None)


_thread_locks_guard = threading.Lock()
_thread_locks = {}


def _thread_lock(book: 'DatabaseBook') -> threading.Lock:
    """Process-wide mutex per book.

    The FileLock guards concurrent *processes*; within one process (mod_wsgi serves a
    book from several threads) its unlink-on-release semantics leave a window in which
    two threads can both believe they hold it, which would drop an assignment."""
    with _thread_locks_guard:
        return _thread_locks.setdefault(book.book, threading.Lock())


class DatabaseBookAssignments:
    """Storage of page_assignments.json, the file of record for a book's assignments."""

    @staticmethod
    def path(book: 'DatabaseBook') -> str:
        return book.local_path(_assignments_file)

    @staticmethod
    def lock(book: 'DatabaseBook') -> FileLock:
        """Inter-process lock guarding read-modify-write cycles on page_assignments.json.

        Not reentrant across FileLock instances: never call mutate() from within mutate()."""
        return FileLock(book.local_path(_assignments_file + '.lock'), timeout=30)

    @staticmethod
    def load(book: 'DatabaseBook') -> BookAssignments:
        path = DatabaseBookAssignments.path(book)
        try:
            with open(path) as f:
                return BookAssignments.from_json(f.read())
        except FileNotFoundError:
            return BookAssignments()
        except Exception as e:
            # a corrupted file must not take the whole book down
            logger.warning('Could not parse {}, treating the book as unassigned'.format(path))
            logger.exception(e)
            return BookAssignments()

    @staticmethod
    def to_file(book: 'DatabaseBook', assignments: BookAssignments):
        for a in assignments.assignments:
            # the two-phase bulk rename can transiently produce duplicates
            a.pages = list(dict.fromkeys(a.pages))
        s = assignments.to_json()
        path = DatabaseBookAssignments.path(book)
        # atomic replace: a reader must never observe a partially written file
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.page_assignments.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(s)
            # mkstemp creates the file 0600 and os.replace preserves that, so without
            # this the file becomes unreadable to every other user — e.g. the Apache
            # worker (www-data), which then 500s on the book.
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def mutate(book: 'DatabaseBook', fn: Callable[[BookAssignments], object]):
        """Locked read-modify-write. Returns whatever fn returned."""
        with _thread_lock(book), DatabaseBookAssignments.lock(book):
            assignments = DatabaseBookAssignments.load(book)
            result = fn(assignments)
            DatabaseBookAssignments.to_file(book, assignments)
            return result


def sort_pages_in_book_order(book: 'DatabaseBook', pages: List[str]) -> List[str]:
    order = {name: i for i, name in enumerate(book.page_names_on_disk())}
    # unknown labels (deleted out of band) keep their relative order at the end
    return sorted(dict.fromkeys(pages), key=lambda p: (order.get(p, len(order)), p))


def rename_page_in_assignments(book: 'DatabaseBook', old_name: str, new_name: str):
    if not os.path.exists(DatabaseBookAssignments.path(book)):
        return  # the overwhelmingly common case: never pay for a lock

    def apply(assignments: BookAssignments):
        for a in assignments.assignments:
            a.pages = [new_name if p == old_name else p for p in a.pages]

    DatabaseBookAssignments.mutate(book, apply)


def remove_page_from_assignments(book: 'DatabaseBook', name: str):
    if not os.path.exists(DatabaseBookAssignments.path(book)):
        return

    def apply(assignments: BookAssignments):
        for a in assignments.assignments:
            a.pages = [p for p in a.pages if p != name]

    DatabaseBookAssignments.mutate(book, apply)


def safe(func):
    """Assignment bookkeeping must never fail the page operation that triggered it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning('Page assignment update {} failed'.format(func.__name__))
            logger.exception(e)
            return None
    return wrapper


safe_rename_page_in_assignments = safe(rename_page_in_assignments)
safe_remove_page_from_assignments = safe(remove_page_from_assignments)
