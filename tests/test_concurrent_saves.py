import json
import os
import threading

from database import DatabasePage
from database.file_formats.pcgts import PcGts
from tests.test_book_documents import (PAGE_A, PAGE_B, TemporaryDemoBookTestCase,
                                       TestBookDocuments)


class TestConcurrentSaves(TemporaryDemoBookTestCase):
    """Several people saving at the same time.

    Saves of *one* page are serialised by the page edit lock, but the files of a book are
    read while other pages are being written: the documents worker parses every page, other
    users view them, tasks run on them. A writer that truncates the file in place therefore
    exposes a window in which a reader sees invalid JSON -- and for pcgts.json that window
    was answered with "corrupt, recreate it" (PagePcGtsView.get), i.e. the page was dropped.
    """

    ITERATIONS = 25

    def _run_concurrently(self, targets):
        errors = []

        def guarded(fn):
            def run():
                try:
                    fn()
                except Exception as e:  # pragma: no cover - only on failure
                    errors.append(e)
            return run

        threads = [threading.Thread(target=guarded(t)) for t in targets]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual([str(e) for e in errors], [])

    def test_readers_never_see_a_half_written_page(self):
        page = self.book.page(PAGE_A)
        path = page.file('pcgts').local_path()
        pcgts = page.pcgts()
        stop = threading.Event()

        def writer():
            try:
                for _ in range(self.ITERATIONS):
                    pcgts.to_file(path)
            finally:
                stop.set()

        def reader():
            while not stop.is_set():
                # the file must always parse: either the previous or the new content
                with open(path) as f:
                    json.load(f)

        self._run_concurrently([writer] + [reader] * 4)

    def test_ten_pages_saved_at_once(self):
        pages = [p for p in self.book.pages() if os.path.exists(p.file('pcgts').local_path())]
        self.assertGreaterEqual(len(pages), 10, 'fixture must offer enough pages')
        pages = pages[:10]
        # parse once up front so the threads only exercise the writing side
        parsed = {p.page: p.pcgts() for p in pages}

        def save(db_page: DatabasePage):
            def run():
                for _ in range(5):
                    parsed[db_page.page].to_file(db_page.file('pcgts').local_path())
            return run

        self._run_concurrently([save(p) for p in pages])

        # every page must still be readable and carry its own content
        for p in pages:
            reloaded = PcGts.from_file(p.file('pcgts'))
            self.assertEqual(reloaded.page.p_id, parsed[p.page].page.p_id)

    def test_concurrent_document_updates_from_many_savers(self):
        """Ten saves of the same book collapse into the background worker, not into ten
        concurrent full recomputations of the book's chant list."""
        import restapi.consumers as consumers
        from database.database_book_documents import DatabaseBookDocuments

        DatabaseBookDocuments.update_book_documents_cached(self.book)
        TestBookDocuments._set_document_starts(self.book.page(PAGE_B), [1])

        self._run_concurrently([lambda: consumers.schedule_book_documents_update(self.book)] * 10)
        for _ in range(100):
            worker = consumers._documents_update_worker
            if worker is None:
                break
            worker.join(timeout=5)
        else:  # pragma: no cover - only on failure
            self.fail('the documents update worker did not finish')

        # the last save is reflected, and the file it wrote is intact
        with open(self.book.local_path('book_documents.json')) as f:
            on_disk = json.load(f)
        full = DatabaseBookDocuments.update_book_documents(self.book)
        self.assertEqual(on_disk['database_documents'], full.database_documents.to_json())
