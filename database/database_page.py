from database.database_book import DatabaseBook, file_name_validator, InvalidFileNameException, FileExistsException
from django.core.exceptions import EmptyResultSet
from database.database_permissions import DatabaseBookPermissionFlag
from typing import Optional
import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from django.contrib.auth.models import User


class DatabasePage:
    def __init__(self, book: DatabaseBook, page: str, skip_validation=False):
        self.book = book
        self.page = page.strip("/")
        if not skip_validation and not file_name_validator.fullmatch(self.page):
            raise InvalidFileNameException(self.page)

        from database.database_page_meta import DatabasePageMeta
        self._meta: Optional[DatabasePageMeta] = None

    def __eq__(self, other):
        return isinstance(other, DatabasePage) and self.book == other.book and self.page == other.page

    def exists(self):
        return os.path.isdir(self.local_path())

    def delete(self):
        if os.path.exists(self.local_path()):
            shutil.rmtree(self.local_path())

    def rename(self, new_name):
        if not file_name_validator.fullmatch(new_name):
            raise InvalidFileNameException(new_name)

        old_path = self.local_path()
        self.page = new_name
        new_path = self.local_path()

        if os.path.exists(new_path):
            raise FileExistsException(new_name, new_path)

        shutil.move(old_path, new_path)

    def file(self, fileId, create_if_not_existing=False):
        from database.database_file import DatabaseFile
        return DatabaseFile(self, fileId, create_if_not_existing)

    def local_file_path(self, f):
        return os.path.join(self.local_path(), f)

    def local_path(self):
        return os.path.join(self.book.local_path('pages'), self.page)

    def remote_path(self):
        return os.path.join(self.book.remote_path(), self.page)

    def pcgts(self, create_if_not_existing=True):
        from database.file_formats.pcgts import PcGts
        return PcGts.from_file(self.file('pcgts', create_if_not_existing))

    def meta(self):
        from database.database_page_meta import DatabasePageMeta
        self._meta = DatabasePageMeta.load(self)
        return self._meta

    def save_meta(self):
        if self._meta:
            self._meta.save(self)

    def is_valid(self):
        if not os.path.exists(self.local_path()):
            return True

        if not os.path.isdir(self.local_path()):
            return False

        return True

    def copy_to(self, database_book: DatabaseBook) -> 'DatabasePage':
        if not database_book.exists():
            raise FileNotFoundError("Database {} not existing".format(database_book.local_path()))

        copy_page = DatabasePage(database_book, self.page)

        if copy_page.exists():
            shutil.rmtree(copy_page.local_path())

        shutil.copytree(self.local_path(), copy_page.local_path())
        return copy_page

    def is_locked(self):
        lock_path = self.local_file_path('.lock')
        if not os.path.exists(lock_path):
            return False

        user = open(lock_path, 'r').read()
        from django.contrib.auth.models import User
        try:
            user = User.objects.get(username=user)
            # check if locked user has sufficient permissions
            if self.book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.WRITE):
                return True
            else:
                # invalid lock, release it
                self.release_lock()
                return False
        except (EmptyResultSet, User.DoesNotExist):
            return False

    def lock_user(self) -> Optional['User']:
        if not self.is_locked():
            return None
        else:
            lock_path = self.local_file_path('.lock')
            with open(lock_path, 'r') as f:
                from django.contrib.auth.models import User
                try:
                    return User.objects.get(username=f.read())
                except (EmptyResultSet, User.DoesNotExist):
                    return None

    def is_locked_by_user(self, user: 'User'):
        lock_path = self.local_file_path('.lock')
        if not os.path.exists(lock_path):
            return False

        from database.database_permissions import DatabaseBookPermissionFlag
        if not self.book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.READ_WRITE):
            return False

        with open(lock_path, 'r') as f:
            return f.read() == user.username

    def lock(self, user: 'User'):
        lock_path = self.local_file_path('.lock')
        with open(lock_path, 'w') as f:
            return f.write(user.username)

    def release_lock(self):
        lock_path = self.local_file_path('.lock')
        if os.path.exists(lock_path):
            os.remove(lock_path)
