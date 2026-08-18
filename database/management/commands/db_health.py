import os
import subprocess

from django.core.management.base import BaseCommand
from django.db import connection

from database.db_errors import db_environment, db_path

# filesystems that cannot back the shared-memory index WAL needs: SQLite then reports
# SQLITE_IOERR_SHMMAP/SHMOPEN, whose message is the generic "disk I/O error"
NO_WAL_FILESYSTEMS = ('9p', 'nfs', 'nfs4', 'cifs', 'smbfs', 'fuse.sshfs', 'virtiofs', 'vboxsf')


def _filesystem_type(path: str) -> str:
    try:
        return subprocess.run(['df', '--output=fstype', path], capture_output=True, text=True,
                              timeout=10).stdout.splitlines()[-1].strip()
    except Exception:
        return 'unknown'


class Command(BaseCommand):
    help = 'Report the health of the SQLite database: paths, permissions, free space, ' \
           'file descriptors, journal mode and integrity. Run this when a request failed ' \
           'with "disk I/O error".'

    def add_arguments(self, parser):
        parser.add_argument('--quick', action='store_true',
                            help='Run PRAGMA quick_check instead of the slower integrity_check')
        parser.add_argument('--set-wal', action='store_true',
                            help='Convert the database to WAL mode. Converting needs exclusive '
                                 'access, so stop the server (and any task worker) first.')

    def handle(self, *args, **options):
        # transaction_mode is only set once the backend has built its connection params
        connection.ensure_connection()
        env = db_environment()
        path = db_path()
        directory = os.path.dirname(path) or '.'
        fstype = _filesystem_type(directory)

        self.stdout.write('database:            {}'.format(path))
        self.stdout.write('exists:              {}'.format(os.path.exists(path)))
        self.stdout.write('size:                {}'.format(_human(env['db_size'])))
        self.stdout.write('-wal:                {}'.format(_human(env['wal_size'])))
        self.stdout.write('-shm:                {}'.format(_human(env['shm_size'])))
        self.stdout.write('directory writable:  {}'.format(env['dir_writable']))
        self.stdout.write('database writable:   {}'.format(env['db_writable']))
        self.stdout.write('side files writable: {}'.format(env['side_files_writable']))
        self.stdout.write('free space:          {}'.format(_human(env['free_bytes'])))
        self.stdout.write('open fds:            {} of {}'.format(env['open_fds'], env['fd_limit']))
        self.stdout.write('filesystem:          {}'.format(fstype))
        self.stdout.write('transaction mode:    {}'.format(
            getattr(connection, 'transaction_mode', None) or 'DEFERRED'))

        for name, owner in _owners(path):
            self.stdout.write('owner {:<14} {}'.format(name + ':', owner))

        with connection.cursor() as cursor:
            for pragma in ('journal_mode', 'synchronous', 'busy_timeout'):
                cursor.execute('PRAGMA {};'.format(pragma))
                self.stdout.write('PRAGMA {:<13} {}'.format(pragma + ':', cursor.fetchone()[0]))

            check = 'quick_check' if options['quick'] else 'integrity_check'
            cursor.execute('PRAGMA {};'.format(check))
            problems = [str(row[0]) for row in cursor.fetchall()]
            ok = problems == ['ok']
            # a damaged database reports one line per page, which is unreadable in a log
            result = 'ok' if ok else '{} problem(s): {}{}'.format(
                len(problems), '; '.join(problems[:5]), ' ...' if len(problems) > 5 else '')
            style = self.style.SUCCESS if ok else self.style.ERROR
            self.stdout.write('PRAGMA {:<13} {}'.format(check + ':', style(result)))

        if options['set_wal']:
            with connection.cursor() as cursor:
                cursor.execute('PRAGMA journal_mode=WAL;')
                row = cursor.fetchone()
                mode = (row[0] if row else '').lower()
            if mode == 'wal':
                self.stdout.write(self.style.SUCCESS('converted to WAL mode'))
            else:
                self.stdout.write(self.style.ERROR(
                    'still in "{}" mode -- another process is holding the database open; '
                    'stop the server and any task worker and try again'.format(mode)))

        from database.models.book_index import BookIndex, PageEditLock, PageIndex
        self.stdout.write('books indexed:       {}'.format(BookIndex.objects.count()))
        self.stdout.write('pages indexed:       {}'.format(PageIndex.objects.count()))
        self.stdout.write('edit locks held:     {}'.format(PageEditLock.objects.count()))

        with connection.cursor() as cursor:
            cursor.execute('PRAGMA journal_mode;')
            journal_mode = cursor.fetchone()[0].lower()
        if journal_mode != 'wal':
            self.stdout.write(self.style.ERROR(
                'The database is in "{}" mode; the server expects WAL (see database/apps.py). '
                'Stop the server and run "manage.py db_health --set-wal".'.format(journal_mode)))

        if fstype in NO_WAL_FILESYSTEMS:
            self.stdout.write(self.style.ERROR(
                'The database is on a {} filesystem, which cannot map the -shm file that WAL '
                'needs. Move it to a local disk (OMMR4ALL_DB_PATH) or disable WAL in '
                'database/apps.py -- otherwise requests fail with "disk I/O error".'.format(fstype)))
        if env['free_bytes'] is not None and env['free_bytes'] < 512 * 1024 ** 2:
            self.stdout.write(self.style.WARNING('Less than 512 MB free on the database volume.'))
        if env['open_fds'] and env['open_fds'] > 0.8 * env['fd_limit']:
            self.stdout.write(self.style.WARNING(
                'Close to the file descriptor limit; SQLite reports an exhausted limit as '
                '"disk I/O error". Raise it with ulimit -n.'))
        if not env['dir_writable'] or env['db_writable'] is False or not env['side_files_writable']:
            self.stdout.write(self.style.ERROR(
                'The database, its -wal/-shm side files or their directory are not writable by '
                'this user -- a frequent leftover of running migrate as root.'))


def _owners(path):
    from pwd import getpwuid
    for candidate in (path, path + '-wal', path + '-shm'):
        if not os.path.exists(candidate):
            continue
        stat = os.stat(candidate)
        try:
            owner = getpwuid(stat.st_uid).pw_name
        except KeyError:
            owner = str(stat.st_uid)
        yield os.path.basename(candidate), '{} (mode {:o})'.format(owner, stat.st_mode & 0o777)


def _human(size):
    if size is None:
        return '-'
    for unit in ('B', 'kB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return '{:.1f} {}'.format(size, unit)
        size /= 1024
