"""Diagnosing and surviving transient SQLite failures.

The database runs in WAL mode (see database/apps.py), which adds a shared-memory index
(`db.sqlite3-shm`) that every connection maps. When that mapping, the WAL file or a
file descriptor becomes unusable, SQLite reports the generic `disk I/O error` and
Django drops the *extended* result code that would name the cause. Two helpers:

- `describe_db_error` recovers that extended code (SQLITE_IOERR_SHMMAP, SQLITE_FULL,
  SQLITE_READONLY_DIRECTORY, ...) and snapshots the environment that decides it.
- `retry_on_db_error` reconnects and retries once, which is what restarting the server
  achieves -- the file is fine, only this connection's mapping is broken.
"""
import logging
import os
import resource
import shutil
from typing import Callable, Optional, TypeVar

from django.conf import settings
from django.db import OperationalError, connection

logger = logging.getLogger(__name__)

T = TypeVar('T')


def db_path() -> str:
    return str(settings.DATABASES['default']['NAME'])


def _sqlite_error_name(exc: BaseException) -> str:
    """Extended SQLite result code of a Django database error, e.g. SQLITE_IOERR_SHMMAP.

    Django wraps the sqlite3 exception, so the code sits on __cause__ (Python 3.11+)."""
    for candidate in (exc, getattr(exc, '__cause__', None), getattr(exc, '__context__', None)):
        name = getattr(candidate, 'sqlite_errorname', None)
        if name:
            return name
    return 'unknown'


def _open_fd_count() -> Optional[int]:
    try:
        return len(os.listdir('/proc/self/fd'))
    except OSError:
        return None


def _size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def db_environment() -> dict:
    """Everything that decides whether SQLite can do its I/O right now."""
    path = db_path()
    directory = os.path.dirname(path) or '.'
    info = {
        'db': path,
        'db_size': _size(path),
        'wal_size': _size(path + '-wal'),
        'shm_size': _size(path + '-shm'),
        'dir_writable': os.access(directory, os.W_OK),
        'db_writable': os.access(path, os.W_OK) if os.path.exists(path) else None,
        'side_files_writable': all(os.access(p, os.W_OK)
                                   for p in (path + '-wal', path + '-shm') if os.path.exists(p)),
        'open_fds': _open_fd_count(),
        'fd_limit': resource.getrlimit(resource.RLIMIT_NOFILE)[0],
    }
    try:
        usage = shutil.disk_usage(directory)
        info['free_bytes'] = usage.free
    except OSError:
        info['free_bytes'] = None
    return info


def describe_db_error(exc: BaseException) -> str:
    """One-line diagnosis of a database failure, for the log."""
    env = db_environment()
    free = env.get('free_bytes')
    return ('{name} (db={db}, free={free}, wal={wal}, shm={shm}, dir_writable={dirw}, '
            'side_files_writable={sidew}, fds={fds}/{limit})').format(
        name=_sqlite_error_name(exc),
        db=env['db'],
        free='{:.1f} GB'.format(free / 1024 ** 3) if free is not None else '?',
        wal='{} B'.format(env['wal_size']) if env['wal_size'] is not None else '-',
        shm='{} B'.format(env['shm_size']) if env['shm_size'] is not None else '-',
        dirw=env['dir_writable'],
        sidew=env['side_files_writable'],
        fds=env['open_fds'] if env['open_fds'] is not None else '?',
        limit=env['fd_limit'],
    )


def is_corruption(exc: BaseException) -> bool:
    """'database disk image is malformed' -- the file is damaged, no retry can help."""
    text = str(exc)
    return 'malformed' in text or 'file is not a database' in text or 'database corrupt' in text


def log_db_failure(log, what: str, exc: BaseException) -> None:
    """One actionable line instead of the same traceback on every request.

    The description already names the extended SQLite code and the state of the file, so a
    stack trace adds nothing -- except for corruption, where the operator needs to be told
    what to do about it."""
    if is_corruption(exc):
        log.error('{} failed -- THE DATABASE FILE IS DAMAGED: {}. Stop the server and run '
                  '"manage.py db_recover --replace", then "manage.py reindex_books".'.format(
                      what, describe_db_error(exc)))
    else:
        log.warning('{} failed: {}'.format(what, describe_db_error(exc)))


def is_lock_contention(exc: BaseException) -> bool:
    """'database is locked' -- another writer held the lock; retrying is the cure."""
    return 'database is locked' in str(exc) or 'database table is locked' in str(exc)


def retry_on_db_error(fn: Callable[[], T], attempts: int = 2) -> T:
    """Run fn, reconnecting once if the database errors out.

    Two failure modes are worth one retry. A SQLITE_IOERR leaves this connection's WAL
    index mapping unusable while the database file itself is fine, so a fresh connection
    fixes it -- exactly what a server restart accomplishes, without the restart. And a
    write that lost a race ("database is locked") simply needs to run again.

    fn must be a whole standalone operation: inside an open atomic block the connection
    must not be dropped and the statements so far would have to be replayed, so the error
    is logged and re-raised for the outer transaction to deal with.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except OperationalError as e:
            contention = is_lock_contention(e)
            if attempt >= attempts or connection.in_atomic_block:
                log = logger.warning if contention else logger.error
                log('Database error{}: {}'.format(
                    '' if attempt < attempts else ' persisted after {} attempts'.format(attempts),
                    describe_db_error(e)))
                raise
            logger.warning('{}, retrying: {}'.format(
                'Write lock contention' if contention else 'Database error, reconnecting',
                describe_db_error(e)))
            if not contention:
                # a contended write only needs a new transaction, not a new connection
                try:
                    connection.close()
                except Exception:
                    logger.exception('Could not close the failed database connection')
