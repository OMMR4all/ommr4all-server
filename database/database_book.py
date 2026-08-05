import os
import logging
from multiprocessing import Pool
from typing import List, TYPE_CHECKING
import ommr4all.settings as settings
import shutil
import re

if TYPE_CHECKING:
    from database.file_formats.performance import LockState
    from database.database_page import DatabasePage


logger = logging.getLogger(__name__)
file_name_validator = re.compile(r'\w+')


def load_pcgts_func(database_page: 'DatabasePage'):
    _ = database_page.pcgts().page
    return database_page


class InvalidFileNameException(Exception):
    def __init__(self, filename):
        super().__init__("Invalid filename {}".format(filename))
        self.filename = filename


class FileExistsException(Exception):
    def __init__(self, filename, path):
        super().__init__("File {} at {} exists".format(filename, path))
        self.path = path
        self.filename = filename


class DatabaseBook:
    @staticmethod
    def list_available() -> List['DatabaseBook']:
        # e.is_dir() already covers the exists/isdir half of is_valid()
        with os.scandir(settings.PRIVATE_MEDIA_ROOT) as it:
            return [DatabaseBook(e.name) for e in it
                    if e.is_dir() and DatabaseBook(e.name, skip_validation=True).is_valid_name()]

    @staticmethod
    def list_available_of_style(notation_style: str) -> List['DatabaseBook']:
        return [b for b in DatabaseBook.list_available() if b.get_meta().notationStyle == notation_style]

    @staticmethod
    def list_available_book_metas():
        return [b.get_meta() for b in DatabaseBook.list_available()]
    
    @staticmethod
    def list_available_book_metas_for_user(user, flag):
        def user_access(b):
            return b.resolve_user_permissions(user).has(flag)
        return [b.get_meta() for b in DatabaseBook.list_available() if user_access(b)]

    @staticmethod
    def list_all_pages_with_lock(locks: List['LockState']) -> List['DatabasePage']:
        out = []
        for b in DatabaseBook.list_available():
            out += b.pages_with_lock(locks)

        return out

    def __init__(self, book: str, skip_validation=False):
        self.book = book.strip('/')
        if not skip_validation and not file_name_validator.fullmatch(self.book):
            raise InvalidFileNameException(self.book)

        self.permissions = None

    def __eq__(self, other):
        return isinstance(other, DatabaseBook) and other.book == self.book

    def page_names_on_disk(self) -> List[str]:
        """Sorted names of the book's page folders.

        scandir instead of listdir + DatabasePage.is_valid(): DirEntry.is_dir() answers
        from the dirent, where is_valid() cost two stat syscalls per page — 39 of the
        42 ms a 650-page book.pages() used to take, on every book-level request."""
        with os.scandir(self.local_path('pages')) as it:
            return sorted(e.name for e in it if e.is_dir())

    def pages(self, load_pcgts = False) -> List['DatabasePage']:
        assert(self.is_valid())
        from database.database_page import DatabasePage

        pages = [DatabasePage(self, p) for p in self.page_names_on_disk()]

        if load_pcgts:
            # forked children must not inherit the SQLite connection or the
            # shared pcgts cache; they load uncached and re-connect lazily
            from django.db import connections
            connections.close_all()

            with Pool() as p:
                pages = list(p.map(load_pcgts_func, iterable=pages))

        return pages

    def pages_with_lock(self, locks: List['LockState']) -> List['DatabasePage']:
        try:
            from database.book_index import pages_with_lock
            return pages_with_lock(self, locks)
        except Exception as e:
            logger.warning('Page index unavailable for {}, scanning page_progress files'.format(self.book))
            logger.exception(e)

        from database.file_formats.performance.pageprogress import Locks
        out = []
        for p in self.pages():
            pp = p.page_progress()
            if all([pp.locked.get(Locks(lock.label), False) == lock.lock for lock in locks]):
                out.append(p)

        return out

    def page(self, page):
        from database.database_page import DatabasePage
        return DatabasePage(self, page)

    def local_default_models_path(self, sub=''):
        return os.path.join(settings.BASE_DIR, 'internal_storage', 'default_models', self.get_meta().notationStyle, sub)

    def local_default_virtual_keyboards_path(self, sub=''):
        return os.path.join(settings.BASE_DIR, 'internal_storage', 'default_virtual_keyboards', sub)

    def local_models_path(self, sub=''):
        return self.local_path(os.path.join('models', sub))

    def local_path(self, sub=''):
        return os.path.join(settings.PRIVATE_MEDIA_ROOT, self.book, sub)

    def remote_path(self):
        return os.path.join(settings.PRIVATE_MEDIA_URL, self.book)

    def is_valid_name(self):
        return file_name_validator.fullmatch(self.book)

    def is_valid(self):
        if not self.is_valid_name():
            return False

        if not os.path.exists(self.local_path()):
            return True

        if not os.path.isdir(self.local_path()):
            return False

        return True

    def exists(self):
        return os.path.exists(self.local_path()) and os.path.isdir(self.local_path())

    def create(self, book_meta):
        if self.exists():
            return True

        if not self.is_valid():
            return False

        os.mkdir(self.local_path())
        os.mkdir(self.local_path('pages'))
        book_meta.to_file(self)
        return True

    def delete(self):
        if os.path.exists(self.local_path()):
            shutil.rmtree(self.local_path())
        from database.book_index import safe_remove_book
        safe_remove_book(self.book)

    def get_meta(self):
        from database.database_book_meta import DatabaseBookMeta
        return DatabaseBookMeta.load(self)

    def mark_updated(self, user=None):
        # best effort: a failed timestamp bump must never fail the actual write
        try:
            from datetime import datetime
            meta = self.get_meta()
            meta.updated = datetime.now()
            meta.updatedBy = getattr(user, 'username', None) or None
            meta.to_file(self)
        except Exception as e:
            logger.warning('Could not update the last-modified timestamp of book {}'.format(self.book))
            logger.exception(e)

    def save_json_to_meta(self, obj: dict):
        from database.database_book_meta import DatabaseBookMeta
        meta = DatabaseBookMeta.from_dict(obj)
        meta.to_file(self)

    def page_names(self) -> List[str]:
        return [p.page for p in self.pages()]

    def get_permissions(self, reload=False):
        from database.database_permissions import DatabaseBookPermissions
        if self.permissions is None or reload:
            self.permissions = DatabaseBookPermissions.load(self)
        return self.permissions

    def resolve_user_permissions(self, user, reload=False):
        if getattr(user, 'is_superuser', False):
            # full access regardless of the file -- skip loading it at all
            from database.database_permissions import BookPermissionFlags
            return BookPermissionFlags.full_access_flags()
        return self.get_permissions(reload).resolve_user_permissions(user)

    def get_or_add_user_permissions(self, user, default=None, reload=False):
        return self.get_permissions(reload).get_or_add_user_permissions(user, default)
