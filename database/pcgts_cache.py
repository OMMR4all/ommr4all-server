"""In-process LRU cache of parsed PcGts files.

Entries are keyed by the pcgts.json path and validated by st_mtime_ns on every
access, so the cache is safe across the spawned task-worker processes (each
process has its own cache but all validate against the shared files).

Cached objects are shared between requests: treat them as READ-ONLY. Access them
via DatabasePage.pcgts_cached(); every code path that mutates and saves a PcGts
must keep using the uncached DatabasePage.pcgts().
"""
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from django.conf import settings

if TYPE_CHECKING:
    from database.database_page import DatabasePage
    from database.file_formats.pcgts import PcGts

_lock = threading.Lock()
_cache: 'OrderedDict[str, tuple]' = OrderedDict()  # path -> (st_mtime_ns, PcGts)


def _max_size() -> int:
    return getattr(settings, 'PCGTS_CACHE_SIZE', 32)


def get(db_page: 'DatabasePage') -> 'PcGts':
    file = db_page.file('pcgts', create_if_not_existing=True)
    path = file.local_path()
    try:
        mtime_ns = os.stat(path).st_mtime_ns
    except OSError:
        return db_page.pcgts()

    with _lock:
        entry = _cache.get(path)
        if entry is not None and entry[0] == mtime_ns:
            _cache.move_to_end(path)
            return entry[1]

    from database.file_formats.pcgts import PcGts
    pcgts = PcGts.from_file(file)

    with _lock:
        _cache[path] = (mtime_ns, pcgts)
        _cache.move_to_end(path)
        while len(_cache) > _max_size():
            _cache.popitem(last=False)

    return pcgts


def invalidate(path: str):
    with _lock:
        _cache.pop(path, None)


def clear():
    with _lock:
        _cache.clear()
