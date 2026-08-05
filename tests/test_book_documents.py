import json
import os
import shutil
import tempfile
import threading
from typing import List
from unittest import TestCase, mock

from asgiref.sync import async_to_sync

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_STORAGE = os.path.join(BASE_DIR, 'tests', 'storage')

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = FIXTURE_STORAGE

from database import DatabaseBook, DatabasePage  # noqa: E402
from database.database_book_documents import DatabaseBookDocuments, PageDocumentFragment  # noqa: E402

# The only fixture pages with a populated lyrics reading order
PAGE_A = 'page_test_monodi_export_001'
PAGE_B = 'page_test_syllable_detection_001'


class TemporaryDemoBookTestCase(TestCase):
    """Base: runs against a throw-away copy (pcgts + color_original only) of the demo
    book so the checked-in fixture storage is never mutated."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_storage = settings.PRIVATE_MEDIA_ROOT
        settings.PRIVATE_MEDIA_ROOT = self._tmp.name
        src_pages = os.path.join(FIXTURE_STORAGE, 'demo', 'pages')
        for page in os.listdir(src_pages):
            src = os.path.join(src_pages, page)
            pcgts = os.path.join(src, 'pcgts.json')
            if not os.path.exists(pcgts):
                continue
            dst = os.path.join(self._tmp.name, 'demo', 'pages', page)
            os.makedirs(dst)
            shutil.copy(pcgts, dst)
            # PcGts.from_json reads the image size from color_original on every load
            # (the preview must exist too, or DatabaseFile tries to recreate the image)
            shutil.copy(os.path.join(src, 'color_original.jpg'), dst)
            shutil.copy(os.path.join(src, 'color_original_preview.jpg'), dst)
        self.book = DatabaseBook('demo')

    def tearDown(self):
        settings.PRIVATE_MEDIA_ROOT = self._orig_storage
        self._tmp.cleanup()


class TestBookDocuments(TemporaryDemoBookTestCase):
    """Tests for the per-page fragment cache behind the chant/document list.

    The fixture pages carry no documentStart flags, so the test marks some
    reading-order lines itself.
    """

    def setUp(self):
        super().setUp()
        # Some fixture pages carry no p_id: it is minted per load until the page is
        # saved once, so persist every page to get stable ids across reloads.
        for db_page in self.book.pages():
            db_page.pcgts().to_file(db_page.file('pcgts').local_path())
        # Three documents: two starting on PAGE_A (the second spans into PAGE_B),
        # one starting mid PAGE_B and running to the end of the book.
        self._set_document_starts(self.book.page(PAGE_A), [0, 2])
        self._set_document_starts(self.book.page(PAGE_B), [3])

    @staticmethod
    def _set_document_starts(db_page: DatabasePage, reading_order_indices: List[int]):
        pcgts = db_page.pcgts()
        for i in reading_order_indices:
            pcgts.page.reading_order.reading_order[i].document_start = True
        pcgts.to_file(db_page.file('pcgts').local_path())

    @staticmethod
    def _touch(db_page: DatabasePage):
        path = db_page.file('pcgts').local_path()
        os.utime(path, times=(os.path.getmtime(path) + 1, os.path.getmtime(path) + 1))

    def test_cached_matches_full_recompute(self):
        cached = DatabaseBookDocuments.update_book_documents_cached(self.book)
        self.assertEqual(len(cached.database_documents.documents), 3)
        full = DatabaseBookDocuments.update_book_documents(self.book)
        self.assertEqual(cached.database_documents.to_json(), full.database_documents.to_json())

    def test_index_serves_current_documents_and_invalidates_on_change(self):
        from database.book_index import get_documents_json
        written = DatabaseBookDocuments.update_book_documents_cached(self.book)

        data = get_documents_json(self.book)
        self.assertIsNotNone(data, 'a freshly written book_documents.json must be served from the index')
        self.assertEqual(data['database_documents'], written.to_json()['database_documents'])

        # any page whose pcgts moved on invalidates the fast path again
        self._touch(self.book.page(PAGE_B))
        self.assertIsNone(get_documents_json(self.book))

    def test_only_changed_page_is_reparsed(self):
        DatabaseBookDocuments.update_book_documents_cached(self.book)
        self._touch(self.book.page(PAGE_B))
        original_extract = PageDocumentFragment.extract
        extracted = []

        def counting_extract(db_page, mtime):
            extracted.append(db_page.page)
            return original_extract(db_page, mtime)

        with mock.patch.object(PageDocumentFragment, 'extract', side_effect=counting_extract):
            DatabaseBookDocuments.update_book_documents_cached(self.book)
        self.assertEqual(extracted, [PAGE_B])

    def test_no_write_when_nothing_changed(self):
        DatabaseBookDocuments.update_book_documents_cached(self.book)
        with mock.patch.object(DatabaseBookDocuments, 'to_file') as to_file, \
                mock.patch.object(PageDocumentFragment, 'extract') as extract:
            d = DatabaseBookDocuments.update_book_documents_cached(self.book)
        to_file.assert_not_called()
        extract.assert_not_called()
        self.assertEqual(len(d.database_documents.documents), 3)

    def test_doc_ids_and_meta_preserved_across_recompute(self):
        from database.file_formats.book.document import DocumentMetaInfos
        d1 = DatabaseBookDocuments.update_book_documents_cached(self.book)
        ids1 = [doc.doc_id for doc in d1.database_documents.documents]
        d1.database_documents.documents[0].document_meta_infos = DocumentMetaInfos(initium='Initium test')
        with DatabaseBookDocuments.lock(self.book):
            d1.to_file(self.book)

        self._touch(self.book.page(PAGE_A))
        d2 = DatabaseBookDocuments.update_book_documents_cached(self.book)
        self.assertEqual([doc.doc_id for doc in d2.database_documents.documents], ids1)
        self.assertEqual(d2.database_documents.documents[0].document_meta_infos.initium, 'Initium test')

    def test_new_start_creates_document_and_keeps_other_ids(self):
        d1 = DatabaseBookDocuments.update_book_documents_cached(self.book)
        ids1 = [doc.doc_id for doc in d1.database_documents.documents]

        self._set_document_starts(self.book.page(PAGE_A), [1])
        d2 = DatabaseBookDocuments.update_book_documents_cached(self.book)
        docs2 = d2.database_documents.documents
        self.assertEqual(len(docs2), 4)
        # the unchanged first and the two later documents keep their ids
        self.assertEqual(docs2[0].doc_id, ids1[0])
        self.assertEqual([docs2[2].doc_id, docs2[3].doc_id], ids1[1:])

    def test_concurrent_updates_do_not_corrupt(self):
        DatabaseBookDocuments.update_book_documents_cached(self.book)
        errors = []

        def worker(db_page):
            try:
                for _ in range(3):
                    self._touch(db_page)
                    DatabaseBookDocuments.update_book_documents_cached(self.book)
            except Exception as e:  # pragma: no cover - only on failure
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(self.book.page(p),))
                   for p in (PAGE_A, PAGE_B) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

        with open(self.book.local_path('book_documents.json')) as f:
            on_disk = json.load(f)  # must never be torn/corrupt
        full = DatabaseBookDocuments.update_book_documents(self.book)
        self.assertEqual(on_disk['database_documents'], full.database_documents.to_json())


class TestBookDocumentsConsumer(TemporaryDemoBookTestCase):
    """Websocket consumer tests against a temp copy of the demo book whose default
    permissions grant READ (mirrors a publicly readable book)."""

    def setUp(self):
        super().setUp()
        from database.database_permissions import DatabaseBookPermissionFlag
        permissions = self.book.get_permissions()
        permissions.permissions.default.set(DatabaseBookPermissionFlag.READ)
        permissions.write()

    @staticmethod
    def _application(user):
        from channels.routing import URLRouter
        from django.urls import path
        from restapi.consumers import BookDocumentsConsumer, TokenAuthMiddleware

        inner = TokenAuthMiddleware(URLRouter([
            path('ws/book/<str:book>/documents/', BookDocumentsConsumer.as_asgi()),
        ]))

        async def app(scope, receive, send):
            return await inner(dict(scope, user=user), receive, send)
        return app

    def test_anonymous_connects_and_receives_change_events(self):
        # default READ on the book admits anonymous users, like the HTTP endpoints
        async_to_sync(self._connect_and_notify)()

    async def _connect_and_notify(self):
        from channels.layers import get_channel_layer
        from channels.testing import WebsocketCommunicator
        from django.contrib.auth.models import AnonymousUser
        from restapi.consumers import book_documents_group

        communicator = WebsocketCommunicator(self._application(AnonymousUser()), '/ws/book/demo/documents/')
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        await get_channel_layer().group_send(book_documents_group('demo'), {'type': 'documents.changed'})
        self.assertEqual(await communicator.receive_json_from(timeout=5), {'event': 'documents_changed'})
        await communicator.disconnect()

    def test_unknown_book_is_rejected(self):
        async_to_sync(self._rejected)('no_such_book')

    def test_book_without_read_permission_is_rejected(self):
        os.makedirs(os.path.join(self._tmp.name, 'private', 'pages'))
        async_to_sync(self._rejected)('private')

    async def _rejected(self, book: str):
        from channels.testing import WebsocketCommunicator
        from django.contrib.auth.models import AnonymousUser

        communicator = WebsocketCommunicator(self._application(AnonymousUser()),
                                             '/ws/book/{}/documents/'.format(book))
        connected, code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4403)
        await communicator.disconnect()
