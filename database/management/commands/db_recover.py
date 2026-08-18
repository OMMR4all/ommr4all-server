"""Salvage a corrupted SQLite database ("database disk image is malformed").

Most of this database is a rebuildable mirror of the storage folder (see
database/book_index.py) -- but the user accounts, book styles and global permissions are
not, so the file is worth recovering rather than recreating. The original is never
modified: it is copied aside first and left in place unless --replace is given.
"""
import os
import shutil
import subprocess
import sqlite3
import time

from django.core.management.base import BaseCommand, CommandError

from database.db_errors import db_path

# a declared column type SQLite treats as text (TEXT/VARCHAR/CHAR/CLOB and friends)
TEXTUAL = ('char', 'text', 'clob')


def decoded_text(value: bytes):
    """Text stays text; only genuinely damaged text is kept as raw bytes.

    A blanket ``text_factory = bytes`` reads a corrupt database without ever raising -- but
    it poisons the recovered copy: every TEXT value is inserted as a BLOB, and afterwards
    ``user.username`` is b'admin' instead of 'admin', which breaks every comparison and
    fails JSON serialisation. Decode per value instead and fall back only where we must.
    """
    try:
        return value.decode()
    except UnicodeDecodeError:
        return value


class Command(BaseCommand):
    help = 'Recover a corrupted SQLite database into a new file. Stop the server first. ' \
           'Afterwards run "manage.py reindex_books" to rebuild the derived book index.'

    def add_arguments(self, parser):
        parser.add_argument('--path', default=None,
                            help='Database to recover (default: the configured database)')
        parser.add_argument('--output', default=None,
                            help='Where to write the recovered database '
                                 '(default: <database>.recovered)')
        parser.add_argument('--repair-text', action='store_true',
                            help='Do not recover: convert text columns that hold BLOBs back '
                                 'to text, in place. Repairs a database salvaged by an older '
                                 'version of this command. Stop the server first.')
        parser.add_argument('--replace', action='store_true',
                            help='Put the recovered database in place of the original '
                                 '(the original is kept as <database>.corrupt-<timestamp>)')

    def handle(self, *args, **options):
        path = options['path'] or db_path()
        if not os.path.exists(path):
            raise CommandError('No database at {}'.format(path))
        if options['repair_text']:
            return self._repair_text(path)
        output = options['output'] or path + '.recovered'
        if os.path.exists(output):
            raise CommandError('{} already exists, remove it or pass --output'.format(output))

        self.stdout.write('recovering {} -> {}'.format(path, output))
        self.stdout.write(self.style.WARNING(
            'Make sure the server and every task worker are stopped; recovering a database '
            'that is still being written produces another damaged copy.'))

        if shutil.which('sqlite3'):
            salvaged = self._recover_with_cli(path, output)
        else:
            self.stdout.write('sqlite3 command not found, salvaging with Python')
            salvaged = self._recover_with_python(path, output)

        for table, count, error in salvaged:
            if error:
                self.stdout.write(self.style.WARNING(
                    '{:<40} {} row(s), then stopped: {}'.format(table, count, error)))
            else:
                self.stdout.write('{:<40} {} row(s)'.format(table, count))

        with sqlite3.connect(output) as recovered:
            result = [row[0] for row in recovered.execute('PRAGMA integrity_check;')]
        if result == ['ok']:
            self.stdout.write(self.style.SUCCESS('recovered database passes integrity_check'))
        else:
            self.stdout.write(self.style.ERROR(
                'the recovered database still reports problems: {}'.format('; '.join(result[:5]))))

        if options['replace']:
            kept = '{}.corrupt-{}'.format(path, time.strftime('%Y%m%d-%H%M%S'))
            shutil.move(path, kept)
            for side in ('-wal', '-shm'):
                # they describe the file we just moved away
                if os.path.exists(path + side):
                    os.remove(path + side)
            shutil.move(output, path)
            self.stdout.write(self.style.SUCCESS(
                'replaced {} (damaged original kept at {})'.format(path, kept)))
        else:
            self.stdout.write('left in place; move it over {} yourself when you are '
                              'satisfied with it'.format(path))

        self.stdout.write(self.style.WARNING(
            'Run "manage.py reindex_books" afterwards to rebuild the book index.'))

    def _repair_text(self, path):
        """Undo a blanket ``text_factory = bytes`` salvage: BLOB -> TEXT, in place.

        Older versions of this command wrote every TEXT value back as a BLOB. The database
        then works well enough to start the server, but every string read out of it is
        bytes: usernames stop matching, and writing a book meta file fails with
        "Object of type bytes is not JSON serializable".
        """
        self.stdout.write(self.style.WARNING(
            'Repairing {} in place. The server and every task worker must be stopped.'.format(path)))
        backup = '{}.before-text-repair-{}'.format(path, time.strftime('%Y%m%d-%H%M%S'))

        connection = sqlite3.connect(path)
        try:
            with sqlite3.connect(backup) as target:
                connection.backup(target)  # consistent snapshot, WAL content included
            self.stdout.write('backup: {}'.format(backup))

            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                if not row[0].startswith('sqlite_')]
            repaired, dropped, failed = [], [], []
            for table in tables:
                for _cid, column, declared, *_rest in connection.execute(
                        'PRAGMA table_info("{}")'.format(table)):
                    if not any(t in (declared or '').lower() for t in TEXTUAL):
                        continue  # leave anything not declared as text alone
                    try:
                        count = self._cast_to_text(connection, table, column)
                    except sqlite3.IntegrityError as e:
                        # the key already exists as text: a server that kept running after
                        # the salvage inserted a second, correct row beside the blob one
                        try:
                            shadows = self._drop_shadow_rows(connection, table, column)
                            count = self._cast_to_text(connection, table, column)
                        except sqlite3.Error as retry:
                            failed.append(('{}.{}'.format(table, column),
                                           '{} ({})'.format(e, retry)))
                            continue
                        if shadows:
                            dropped.append((table, shadows))
                    except sqlite3.Error as e:
                        failed.append(('{}.{}'.format(table, column), str(e)))
                        continue
                    if count > 0:
                        repaired.append(('{}.{}'.format(table, column), count))
            connection.commit()
            # comparisons changed, so the b-trees keyed on those values must be rebuilt
            connection.execute('REINDEX')
            connection.commit()
            orphans = ['{} row {} -> {}'.format(table, rowid, parent) for table, rowid, parent, _fk
                       in connection.execute('PRAGMA foreign_key_check')]
        finally:
            connection.close()

        for name, count in repaired:
            self.stdout.write('{:<50} {} value(s)'.format(name, count))
        for table, count in dropped:
            self.stdout.write(self.style.WARNING(
                '{:<50} {} duplicate row(s) dropped'.format(table, count)))
        for name, error in failed:
            self.stdout.write(self.style.ERROR('{:<50} not repaired: {}'.format(name, error)))
        if orphans:
            self.stdout.write(self.style.WARNING(
                'foreign key check: {}'.format('; '.join(orphans[:5]))))
        if not repaired and not failed:
            self.stdout.write(self.style.SUCCESS(
                'nothing to repair, no text column holds a BLOB'))
        else:
            self.stdout.write(self.style.SUCCESS(
                'repaired {} column(s)'.format(len(repaired))))
        if failed:
            self.stdout.write(self.style.ERROR(
                'Some columns could not be converted (usually a value that would now collide '
                'with a row written after the salvage). The database before the repair is '
                'kept at {}.'.format(backup)))

    @staticmethod
    def _cast_to_text(connection, table, column):
        cursor = connection.execute(
            'UPDATE "{0}" SET "{1}" = CAST("{1}" AS TEXT) '
            "WHERE typeof(\"{1}\") = 'blob'".format(table, column))
        return cursor.rowcount

    @staticmethod
    def _unique_keys(connection, table, column):
        """The unique/primary keys of `table` that `column` takes part in."""
        for _seq, index, unique, *_rest in connection.execute(
                'PRAGMA index_list("{}")'.format(table)):
            if not unique:
                continue
            columns = [row[2] for row in connection.execute(
                'PRAGMA index_info("{}")'.format(index)) if row[2] is not None]
            if column in columns:
                yield columns

    def _drop_shadow_rows(self, connection, table, column):
        """Delete blob rows whose text twin already exists under a unique key.

        SQLite treats b'demo' and 'demo' as different keys, so a server still running after
        a botched salvage inserted a second, correct row beside every blob one instead of
        updating it. Those two rows are the same logical row; the text one is the live one
        and the blob one only blocks the conversion. Nothing is lost -- this whole index is
        rebuilt by "manage.py reindex_books" anyway."""
        total = 0
        for columns in self._unique_keys(connection, table, column):
            match = ' AND '.join('CAST(a."{0}" AS TEXT) IS CAST(b."{0}" AS TEXT)'.format(c)
                                 for c in columns)
            cursor = connection.execute(
                'DELETE FROM "{table}" WHERE rowid IN ('
                '  SELECT a.rowid FROM "{table}" a, "{table}" b'
                '  WHERE a.rowid <> b.rowid AND {match}'
                "    AND typeof(a.\"{column}\") = 'blob'"
                "    AND typeof(b.\"{column}\") <> 'blob')".format(
                    table=table, column=column, match=match))
            total += cursor.rowcount
        return total

    def _recover_with_cli(self, path, output):
        """sqlite3 .recover -- reads the b-tree directly and skips what it cannot parse."""
        dump = subprocess.run(['sqlite3', path, '.recover'], capture_output=True, text=True)
        if dump.returncode != 0 and not dump.stdout:
            raise CommandError('sqlite3 .recover failed: {}'.format(dump.stderr.strip()))
        if dump.stderr.strip():
            self.stdout.write(self.style.WARNING(dump.stderr.strip()))
        restore = subprocess.run(['sqlite3', output], input=dump.stdout, capture_output=True,
                                 text=True)
        if restore.returncode != 0:
            raise CommandError('rebuilding {} failed: {}'.format(output, restore.stderr.strip()))
        return self._table_counts(output)

    def _recover_with_python(self, path, output):
        """Copy schema and rows table by table, keeping whatever reads cleanly.

        Poor man's .recover: a corrupt page aborts one table's copy, not the whole file."""
        source = sqlite3.connect(path)
        source.text_factory = decoded_text
        target = sqlite3.connect(output)
        salvaged = []
        try:
            schema = list(source.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
            for _type, name, sql in schema:
                try:
                    target.execute(sql.decode() if isinstance(sql, bytes) else sql)
                except sqlite3.Error as e:
                    self.stdout.write(self.style.WARNING(
                        'could not recreate {}: {}'.format(name, e)))

            for _type, name, _sql in [s for s in schema if s[0] == b'table' or s[0] == 'table']:
                table = name.decode() if isinstance(name, bytes) else name
                count, error = 0, None
                try:
                    cursor = source.execute('SELECT * FROM "{}"'.format(table))
                    columns = len(cursor.description)
                    placeholders = ','.join('?' * columns)
                    while True:
                        try:
                            row = cursor.fetchone()
                        except sqlite3.DatabaseError as e:
                            error = str(e)
                            break
                        if row is None:
                            break
                        target.execute('INSERT OR IGNORE INTO "{}" VALUES ({})'.format(
                            table, placeholders), row)
                        count += 1
                except sqlite3.DatabaseError as e:
                    error = str(e)
                target.commit()
                salvaged.append((table, count, error))
        finally:
            source.close()
            target.close()
        return salvaged

    @staticmethod
    def _table_counts(path):
        connection = sqlite3.connect(path)
        try:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            counts = []
            for table in tables:
                try:
                    count = connection.execute('SELECT count(*) FROM "{}"'.format(table)).fetchone()[0]
                    counts.append((table, count, None))
                except sqlite3.Error as e:
                    counts.append((table, 0, str(e)))
            return counts
        finally:
            connection.close()
