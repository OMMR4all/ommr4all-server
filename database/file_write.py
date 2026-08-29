"""Atomic writes for the JSON files that make up the storage.

The storage folder is the source of truth and is read concurrently while it is written:
several people edit different pages of the same book, the documents worker parses the pages
of the book in the background, tasks read pages while they run. A plain `open(path, 'w')`
truncates the file first, so a reader arriving in that window sees a half-written file —
which for pcgts.json used to be answered with "corrupt, recreate it" (PagePcGtsView.get),
i.e. losing the page. Writing to a temporary file in the same directory and renaming it over
the target makes the switch atomic: a reader sees either the old or the new content.
"""
import os
import tempfile


def write_text_atomic(path: str, content: str, mode: int = 0o644):
    """Replace `path` with `content` in one step, leaving no partially written state."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                    prefix='.' + os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        # mkstemp creates the file 0600 and os.replace preserves that, so without this the
        # file becomes unreadable to every other user -- e.g. the Apache worker (www-data),
        # which then fails on the next read.
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
