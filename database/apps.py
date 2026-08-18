#import torch
import logging

from django.apps import AppConfig
from django.db.backends.signals import connection_created
#try:
#    # torch fork problem workaround
#    torch.set_num_threads(1)
#except RuntimeError:
#    pass

logger = logging.getLogger(__name__)


def set_sqlite_pragmas(sender, connection, **kwargs):
    """Put every SQLite connection into WAL mode.

    Several processes write this database concurrently (the Apache/mod_wsgi
    workers, the spawned task workers and, in the Docker setup, daphne). With the
    default rollback journal a reader blocks the writer and vice versa, so a page
    save could be starved by a concurrent book (re-)index until the 20s busy
    timeout in settings.DATABASES ran out -- the "database is locked" the book
    index used to swallow. Under WAL only writer-vs-writer serialises, and
    synchronous=NORMAL drops the per-commit fsync (safe in WAL: a crash can lose
    the last transactions but cannot corrupt the file), which keeps the write lock
    held for microseconds instead of milliseconds.

    Best effort on purpose: if the pragma itself hits a lock, the connection is
    still perfectly usable with the previous journal mode.
    """
    if connection.vendor != 'sqlite':
        return
    try:
        with connection.cursor() as cursor:
            # the pragma returns the mode actually in effect: converting a database that
            # other connections are using fails with SQLITE_BUSY and silently leaves it in
            # rollback-journal mode, which must not be paired with synchronous=NORMAL --
            # that combination lets a reader observe a torn write
            cursor.execute('PRAGMA journal_mode=WAL;')
            row = cursor.fetchone()
            mode = (row[0] if row else '').lower()
            if mode == 'wal':
                cursor.execute('PRAGMA synchronous=NORMAL;')
            else:
                logger.warning(
                    'SQLite is in "{}" mode, not WAL (another connection likely held the '
                    'database while it was opened). Keeping synchronous=FULL. Run '
                    '"manage.py db_health --set-wal" with the server stopped to convert '
                    'it.'.format(mode or 'unknown'))
    except Exception as e:
        logger.warning('Could not apply the SQLite pragmas: {}'.format(e))


class DatabaseConfig(AppConfig):
    name = 'database'

    def ready(self):
        connection_created.connect(set_sqlite_pragmas, dispatch_uid='ommr4all_sqlite_pragmas')
