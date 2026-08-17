from json import JSONDecodeError

from database.database_book import DatabaseBook, file_name_validator, InvalidFileNameException, FileExistsException
from database.database_permissions import DatabaseBookPermissionFlag
from typing import Optional
import os
import shutil
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)


def edit_lock_is_stale(lock) -> bool:
    """A PageEditLock older than settings.PAGE_EDIT_LOCK_TTL_HOURS (0 disables)."""
    from datetime import timedelta
    from django.conf import settings
    from django.utils import timezone
    ttl_hours = getattr(settings, 'PAGE_EDIT_LOCK_TTL_HOURS', 12)
    if not ttl_hours:
        return False
    return lock.acquired_at < timezone.now() - timedelta(hours=ttl_hours)

if TYPE_CHECKING:
    from django.contrib.auth.models import User
    from database.file_formats.pcgts import PcGts
    from database.database_page_meta import DatabasePageMeta
    from database.file_formats.performance.pageprogress import PageProgress
    from database.file_formats.performance.statistics import Statistics


class DatabasePage:
    def __init__(self, book: DatabaseBook, page: str, skip_validation=False,
                 pcgts: Optional['PcGts'] = None,
                 meta: Optional['DatabasePageMeta'] = None,
                 page_progress: Optional['PageProgress'] = None,
                 page_statistics: Optional['Statistics'] = None,
                 ):
        self.book = book
        self.page = page.strip("/")
        if not skip_validation and not file_name_validator.fullmatch(self.page):
            raise InvalidFileNameException(self.page)

        self._meta: Optional['DatabasePageMeta'] = meta
        self._pcgts: Optional['PcGts'] = pcgts
        self._page_progress: Optional['PageProgress'] = page_progress
        self._page_statistics: Optional['Statistics'] = page_statistics

    def __eq__(self, other):
        return isinstance(other, DatabasePage) and self.book == other.book and self.page == other.page

    def __hash__(self):
        return hash(self.local_path())

    def exists(self):
        return os.path.isdir(self.local_path())

    def delete(self):
        from database import pcgts_cache
        pcgts_cache.invalidate(self.local_file_path('pcgts.json'))
        if os.path.exists(self.local_path()):
            shutil.rmtree(self.local_path())
        from database.book_index import safe_remove_page
        safe_remove_page(self.book.book, self.page)

    def rename(self, new_name):
        if not file_name_validator.fullmatch(new_name):
            raise InvalidFileNameException(new_name)

        old_path = self.local_path()
        old_name = self.page
        self.page = new_name
        new_path = self.local_path()

        if os.path.exists(new_path):
            self.page = old_name
            raise FileExistsException(new_name, new_path)

        from database import pcgts_cache
        pcgts_cache.invalidate(os.path.join(old_path, 'pcgts.json'))

        shutil.move(old_path, new_path)

        from database.book_index import safe_remove_page, safe_index_page
        safe_remove_page(self.book.book, old_name)
        safe_index_page(self)

    def file(self, fileId, create_if_not_existing=False):
        from database.database_file import DatabaseFile
        return DatabaseFile(self, fileId, create_if_not_existing)

    def local_file_path(self, f):
        return os.path.join(self.local_path(), f)

    def local_path(self):
        return os.path.join(self.book.local_path('pages'), self.page)

    def remote_path(self):
        return os.path.join(self.book.remote_path(), self.page)

    def page_statistics(self) -> 'Statistics':
        if not self._page_statistics:
            from database.file_formats.performance.statistics import Statistics
            file = self.file('statistics', create_if_not_existing=True)
            try:
                self._page_statistics = Statistics.from_json_file(file.local_path())
            except JSONDecodeError as e:
                logging.error(e)
                file.delete()
                file.create()
                self._page_statistics = Statistics.from_json_file(file.local_path())

        return self._page_statistics

    def set_page_statistics(self, page_statistics: 'Statistics'):
        self._page_statistics = page_statistics

    def save_page_statistics(self):
        if not self._page_statistics:
            return

        self._page_statistics.to_json_file(self.file('statistics').local_path())
        logger.debug('Successfully saved page statistics file to {}'.format(self.file('statistics').local_path()))

    def page_progress(self) -> 'PageProgress':
        if not self._page_progress:
            from database.file_formats.performance.pageprogress import PageProgress
            self._page_progress = PageProgress.from_json_file(
                self.file('page_progress', create_if_not_existing=True).local_path())

        return self._page_progress

    def set_page_progress(self, page_progress: 'PageProgress'):
        self._page_progress = page_progress

    def has_page_progress(self) -> bool:
        """True if the progress is already loaded, i.e. page_progress() will not hit disk."""
        return self._page_progress is not None

    def save_page_progress(self, user=None):
        if not self._page_progress:
            return

        self._page_progress.to_json_file(self.file('page_progress').local_path())
        logger.debug('Successfully saved page progress file to {}'.format(self.file('page_progress').local_path()))
        self.mark_updated(user)

    def pcgts(self, create_if_not_existing=True) -> 'PcGts':
        if not self._pcgts:
            from database.file_formats.pcgts import PcGts
            self._pcgts = PcGts.from_file(self.file('pcgts', create_if_not_existing))
        return self._pcgts

    def pcgts_cached(self) -> 'PcGts':
        """Shared read-only PcGts from the in-process cache (mtime-validated).

        Only for read paths: the returned object graph is shared between requests,
        so it must not be mutated or saved. Writers use pcgts()."""
        if not self._pcgts:
            from database import pcgts_cache
            self._pcgts = pcgts_cache.get(self)
        return self._pcgts

    def pcgts_from_dict(self, d: dict) -> 'PcGts':
        from database.file_formats.pcgts import PcGts
        self._pcgts = PcGts.from_json(d, self)
        return self._pcgts

    def meta(self) -> 'DatabasePageMeta':
        if not self._meta:
            from database.database_page_meta import DatabasePageMeta
            self._meta = DatabasePageMeta.load(self)
        return self._meta

    def save_meta(self):
        if self._meta:
            self._meta.save(self)

    def mark_updated(self, user=None, propagate=True):
        # best effort: a failed timestamp bump must never fail the actual write
        try:
            from datetime import datetime
            meta = self.meta()
            meta.updated = datetime.now()
            meta.updatedBy = getattr(user, 'username', None) or None
            self.save_meta()
        except Exception as e:
            logger.warning('Could not update the last-modified timestamp of page {}/{}'.format(
                self.book.book, self.page))
            logger.exception(e)

        if propagate:
            self.book.mark_updated(user)

        from database.book_index import safe_index_page
        safe_index_page(self)

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

    def _edit_lock(self):
        from database.models.book_index import PageEditLock
        lock = PageEditLock.objects \
            .filter(page__book__name=self.book.book, page__name=self.page) \
            .select_related('user').first()
        if lock is not None and edit_lock_is_stale(lock):
            # abandoned session (browser crash, closed tab): auto-expire
            lock.delete()
            return None
        return lock

    def is_locked(self):
        lock = self._edit_lock()
        if lock is None:
            return False

        # check if locked user has sufficient permissions
        if self.book.resolve_user_permissions(lock.user).has(DatabaseBookPermissionFlag.WRITE):
            return True
        else:
            # invalid lock, release it
            self.release_lock()
            return False

    def lock_user(self) -> Optional['User']:
        if not self.is_locked():
            return None
        lock = self._edit_lock()
        return lock.user if lock else None

    def is_locked_by_user(self, user: 'User'):
        lock = self._edit_lock()
        if lock is None:
            return False

        from database.database_permissions import DatabaseBookPermissionFlag
        if not self.book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.READ_WRITE):
            return False

        return lock.user_id == user.id

    def lock(self, user: 'User'):
        from django.db import transaction
        from database.book_index import index_page
        from database.models.book_index import PageEditLock
        with transaction.atomic():
            row = index_page(self)
            PageEditLock.objects.update_or_create(page=row, defaults={'user': user})

    def release_lock(self):
        from database.models.book_index import PageEditLock
        PageEditLock.objects.filter(page__book__name=self.book.book, page__name=self.page).delete()
        # transition: also remove a leftover pre-DB lock file
        lock_path = self.local_file_path('.lock')
        if os.path.exists(lock_path):
            os.remove(lock_path)
