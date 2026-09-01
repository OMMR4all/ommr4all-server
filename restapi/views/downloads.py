"""Streaming zip downloads for book and page exports.

The exports used to be assembled in an `io.BytesIO` and handed to a `FileResponse`,
which meant the *whole* archive — for a backup that is every image of every page —
sat in the single mod_wsgi daemon process before the first byte reached the client.
Peak RSS matched the size of the book and glibc never handed the arena back, so the
process stayed large for the rest of its life.

Here the archive is instead streamed straight off disk. Every entry is stored
uncompressed, which lets `zipstream-ng` compute the exact archive length up front,
so the response carries a real `Content-Length` and the browser can show a
percentage and an ETA while it writes the file to disk.
"""

import logging
import os
from typing import Iterable, Iterator, List, NamedTuple, Optional

from django.http import StreamingHttpResponse
from django.utils.http import content_disposition_header
from zipstream import ZipStream

from database import DatabaseBook, DatabasePage
from database.database_file import regenerable_page_filenames

logger = logging.getLogger(__name__)

# 64 KiB is a compromise between syscall count and the amount of data that a
# cancelled download keeps alive for one more iteration.
READ_CHUNK_SIZE = 1 << 16


class ZipEntry(NamedTuple):
    path: str      # absolute path on disk
    arcname: str   # name inside the archive
    size: int      # size in bytes, as stat'ed while enumerating


def _read_exactly(path: str, size: int, chunk_size: int = READ_CHUNK_SIZE) -> Iterator[bytes]:
    """Yield exactly `size` bytes of `path`, padding if the file shrank meanwhile.

    The entry sizes are stat'ed while enumerating but the bytes are only read once
    the client is already downloading. A background OMR task rewriting a pcgts.json
    in between would otherwise make the streamed length disagree with the promised
    Content-Length, which aborts the transfer and leaves the user with a truncated
    backup. Padding keeps the archive well-formed; a short read only affects the one
    file that was being written at that moment.
    """
    left = size
    try:
        with open(path, 'rb') as f:
            while left > 0:
                buffer = f.read(min(chunk_size, left))
                if not buffer:
                    break
                left -= len(buffer)
                yield buffer
    except OSError as e:
        logger.warning('Could not read {} while streaming an export: {}'.format(path, e))

    if left > 0:
        logger.warning('{} shrank while streaming an export, padding {} bytes'.format(path, left))
        while left > 0:
            n = min(chunk_size, left)
            left -= n
            yield b'\0' * n


def entry_for(path: str, arcname: str) -> Optional[ZipEntry]:
    """A single file as an archive entry, or None if it vanished."""
    try:
        return ZipEntry(path, arcname, os.path.getsize(path))
    except OSError:
        return None


def backup_entries(book: DatabaseBook, full: bool = False) -> List[ZipEntry]:
    """Everything of a book that is worth archiving, in on-disk order.

    Skipped are the per page edit histories (`pcgts_backup.zip` and friends, which
    dominate the size of a book folder), transient lock files, and — unless `full`
    is requested — every image that `DatabaseFile.create()` can recompute from
    color_original. What remains is the original scans, the annotations, and the
    book's own trained models.
    """
    skip = set() if full else regenerable_page_filenames()
    root_path = book.local_path()
    entries = []
    for root, dirs, files in os.walk(root_path):
        for file in sorted(files):
            if file.endswith('.zip') or file.endswith('.lock') or file in skip:
                continue

            path = os.path.join(root, file)
            # keep the book folder as the first path component: BooksImportView
            # derives the book to restore into from the first entry's top directory
            entry = entry_for(path, os.path.join(book.book, os.path.relpath(path, root_path)))
            if entry:
                entries.append(entry)

    return entries


def annotation_entries(pages: List[DatabasePage]) -> List[ZipEntry]:
    """The full annotations grouped by file type, one directory per type."""
    file_names = ['color_original', 'color_norm_x2', 'binary_norm_x2', 'pcgts', 'meta']
    entries = []
    for page in pages:
        files = [page.file(f) for f in file_names]
        if any([not f.exists() for f in files]):
            continue

        for file, fn in zip(files, file_names):
            entry = entry_for(file.local_path(), os.path.join(fn, page.page + file.ext()))
            if entry:
                entries.append(entry)

    return entries


def original_image_entries(book: DatabaseBook, pages: List[DatabasePage]) -> List[ZipEntry]:
    entries = []
    for page in pages:
        file = page.file('color_original')
        if not file.exists():
            continue

        entry = entry_for(file.local_path(), os.path.join(book.book, page.page + file.ext()))
        if entry:
            entries.append(entry)

    return entries


def sanitize_filename(name: str) -> str:
    """Book titles are free text; keep them out of the path and the header syntax."""
    name = ''.join(c for c in name if c.isprintable() and c not in '"\\/')
    name = name.replace(os.sep, '').strip()
    return name or 'book'


def stream_zip_response(entries: Iterable[ZipEntry], filename: str) -> StreamingHttpResponse:
    """A zip of `entries`, streamed from disk with an exact Content-Length."""
    zs = ZipStream(sized=True)  # sized implies ZIP_STORED, which is what makes len() exact
    for entry in entries:
        zs.add(_read_exactly(entry.path, entry.size), entry.arcname, size=entry.size)

    response = StreamingHttpResponse(zs, content_type='application/zip')
    response['Content-Length'] = str(len(zs))
    response['Content-Disposition'] = content_disposition_header(True, filename)
    response['Cache-Control'] = 'no-store'
    # FileResponse exposes this and the export tests use it in assertion messages
    response.filename = filename
    return response
