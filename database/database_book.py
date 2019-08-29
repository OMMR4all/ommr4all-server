import os
import logging
from typing import List, TYPE_CHECKING
import ommr4all.settings as settings
import shutil
import re

if TYPE_CHECKING:
    from database.file_formats.performance import LockState
    from database.database_page import DatabasePage


logger = logging.getLogger(__name__)
file_name_validator = re.compile(r'\w+')


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
        return [DatabaseBook(name) for name in os.listdir(settings.PRIVATE_MEDIA_ROOT) if DatabaseBook(name, skip_validation=True).is_valid()]

    @staticmethod
    def list_available_of_style(notation_style: str) -> List['DatabaseBook']:
        return [b for b in DatabaseBook.list_available() if b.get_meta().notationStyle == notation_style]

    @staticmethod
    def list_available_book_metas():
        return [b.get_meta() for b in DatabaseBook.list_available()]

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

    def pages(self) -> List['DatabasePage']:
        assert(self.is_valid())
        from database.database_page import DatabasePage

        pages = [DatabasePage(self, p) for p in sorted(os.listdir(self.local_path('pages')))]
        return [p for p in pages if p.is_valid()]

    def pages_with_lock(self, locks: List['LockState']) -> List['DatabasePage']:
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

    def get_meta(self):
        from database.database_book_meta import DatabaseBookMeta
        return DatabaseBookMeta.load(self)

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
        return self.get_permissions(reload).resolve_user_permissions(user)

    def get_or_add_user_permissions(self, user, default=None, reload=False):
        return self.get_permissions(reload).get_or_add_user_permissions(user, default)
