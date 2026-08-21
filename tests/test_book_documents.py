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


class TestDocumentSpans(TestBookDocuments):
    """The assembled span must match the lines the document actually covers."""

    def _reading_order(self, page_name):
        return [line.id for line in self.book.page(page_name).pcgts().page.reading_order.reading_order]

    def test_span_matches_the_lines_of_the_document(self):
        for doc in DatabaseBookDocuments.update_book_documents_cached(self.book).database_documents.documents:
            lines = doc.get_page_line_of_document(self.book)
            self.assertTrue(lines, 'document {} covers no line'.format(doc.doc_id))
            first_line, first_page = lines[0]
            last_line, last_page = lines[-1]
            ro_last = self._reading_order(last_page.page)

            self.assertEqual(doc.start.page_name, first_page.page)
            self.assertEqual(doc.start.row, self._reading_order(first_page.page).index(first_line.id) + 1)
            # end.page_name/row describe the last line taken, not the exclusive cursor
            self.assertEqual(doc.end.page_name, last_page.page)
            self.assertEqual(doc.end.row, ro_last.index(last_line.id) + 1)
            self.assertLessEqual(doc.end.row, len(ro_last))
            # no page the document does not reach
            self.assertEqual(doc.pages_names, sorted(set(page.page for _, page in lines), key=doc.pages_names.index))
            self.assertEqual(doc.pages_names[-1], last_page.page)

    def test_last_document_runs_to_the_end_of_the_book(self):
        docs = DatabaseBookDocuments.update_book_documents_cached(self.book).database_documents.documents
        lines = docs[-1].get_page_line_of_document(self.book)
        last_line, last_page = lines[-1]
        self.assertEqual(last_line.id, self._reading_order(last_page.page)[-1])
        self.assertEqual(docs[-1].end.line_id, '', 'the last document has no successor to stop at')

    def test_document_ending_at_a_page_break_keeps_no_extra_page(self):
        # a start on the very first line of PAGE_B ends the previous document on PAGE_A
        self._set_document_starts(self.book.page(PAGE_B), [0])
        docs = DatabaseBookDocuments.update_book_documents_cached(self.book).database_documents.documents
        ending_at_break = [d for d in docs if d.start.page_name == PAGE_A and d.end.page_name == PAGE_A]
        self.assertTrue(ending_at_break)
        for doc in ending_at_break:
            self.assertEqual(doc.pages_names, [PAGE_A])
            self.assertLessEqual(doc.end.row, len(self._reading_order(PAGE_A)))

    def test_stale_version_forces_one_reassembly(self):
        from database.book_index import get_documents_json
        DatabaseBookDocuments.update_book_documents_cached(self.book)
        self.assertIsNotNone(get_documents_json(self.book))

        path = self.book.local_path('book_documents.json')
        with open(path) as f:
            stored = json.load(f)
        stored.pop('version')
        mtime = os.path.getmtime(path)
        with open(path, 'w') as f:
            json.dump(stored, f)
        os.utime(path, times=(mtime, mtime))  # only the version, not the file, changed
        from database.book_index import safe_index_documents
        safe_index_documents(self.book)

        self.assertIsNone(get_documents_json(self.book), 'documents of an older format must not be served')
        DatabaseBookDocuments.update_book_documents_cached(self.book)
        self.assertIsNotNone(get_documents_json(self.book))
        # and it settles: no further rewrite
        with mock.patch.object(DatabaseBookDocuments, 'to_file') as to_file:
            DatabaseBookDocuments.update_book_documents_cached(self.book)
        to_file.assert_not_called()


class TestDocumentExport(TestBookDocuments):
    """The per-document exports must contain the document and nothing else."""

    def _documents(self):
        return DatabaseBookDocuments.update_book_documents_cached(self.book).database_documents.documents

    @staticmethod
    def _syllables(notes):
        out = []

        def rec(d):
            if isinstance(d, dict):
                if d.get('kind') == 'Syllable':
                    out.append(d.get('text', ''))
                for v in d.values():
                    rec(v)
            elif isinstance(d, list):
                for v in d:
                    rec(v)

        rec(notes)
        return out

    @staticmethod
    def _letters(text):
        return ''.join(text.replace('-', '').split()).lower()

    def test_monodi_json_covers_only_the_document(self):
        from database.file_formats.exporter.document_export import monodi_json_of_document
        for doc in self._documents():
            payload = monodi_json_of_document(self.book, doc, 'editor')
            self.assertEqual(set(payload.keys()), {'document', 'notes'})
            self.assertEqual(payload['document']['rowstart'], str(doc.start.row))
            self.assertEqual(payload['document']['additionalData']['Endzeile'], str(doc.end.row))

            exported = self._letters(' '.join(self._syllables(payload['notes'])))
            expected = self._letters(doc.get_text_of_document(self.book))
            # every exported letter comes from the document, in order: nothing from the
            # lines above the start or below the end may leak in
            it = iter(expected)
            self.assertTrue(all(c in it for c in exported),
                            'export of {} contains text outside the document'.format(doc.doc_id))

    def test_mei_is_trimmed_to_the_document(self):
        from lxml import etree
        from database.file_formats.exporter.document_export import (
            document_line_ids_by_page, mei_files_of_document, pcgts_of_document,
        )
        from database.file_formats.exporter.mei.pcgts_to_mei4_exporter import PcgtsToMeiConverter
        ns = '{http://www.music-encoding.org/ns/mei}'
        trimmed_somewhere = False
        for doc in self._documents():
            line_ids = document_line_ids_by_page(self.book, doc)
            files = mei_files_of_document(self.book, doc)
            self.assertEqual([name for name, _ in files], doc.pages_names)
            for (page_name, xml), pcgts in zip(files, pcgts_of_document(self.book, doc)):
                staves = etree.fromstring(xml.encode()).findall('.//%sstaff' % ns)
                expected = {line.id for line in
                            (pcgts.page.closest_music_line_to_text_line(t)
                             for t in pcgts.page.reading_order.reading_order
                             if t.id in set(line_ids[page_name]))
                            if line is not None}
                self.assertEqual(len(staves), len(expected))
                whole_page = etree.fromstring(PcgtsToMeiConverter(pcgts).to_string().encode())
                trimmed_somewhere |= len(staves) < len(whole_page.findall('.//%sstaff' % ns))
        self.assertTrue(trimmed_somewhere, 'no page was trimmed relative to the whole-page MEI')

    def test_midi_plays_only_the_document(self):
        from database.file_formats.book.document import staves_of_document_lines, symbols_of_document_staff
        from database.file_formats.exporter.document_export import pcgts_of_document
        from database.file_formats.exporter.midi.simple_midi import SimpleMidiExporter
        smaller_somewhere = False
        for doc in self._documents():
            pcgts = pcgts_of_document(self.book, doc)
            sequence = SimpleMidiExporter(pcgts).generate_note_sequence(document=doc)
            whole_pages = SimpleMidiExporter(pcgts).generate_note_sequence()

            # the notes of the staves the document's lyric lines sit on, and no others
            expected = 0
            by_page = {}
            for line, page in doc.get_lines_of_pcgts(pcgts):
                by_page.setdefault(page.p_id, (page, []))[1].append(line)
            for page, lines in by_page.values():
                ids = {line.id for line in lines}
                for staff in staves_of_document_lines(page, lines):
                    expected += sum(1 for s in symbols_of_document_staff(page, staff, ids)
                                    if s.symbol_type == s.symbol_type.NOTE)

            self.assertEqual(len(sequence['notes']), expected)
            self.assertLessEqual(len(sequence['notes']), len(whole_pages['notes']))
            self.assertEqual(sequence['totalTime'], len(sequence['notes']) * 0.5)
            smaller_somewhere |= len(sequence['notes']) < len(whole_pages['notes'])
        self.assertTrue(smaller_somewhere, 'no document played fewer notes than its whole pages')

    def test_monodi_metadata_identifies_the_book_not_mulhouse(self):
        from database.database_book_meta import DatabaseBookMeta
        from database.file_formats.exporter.document_export import monodi_json_of_document
        doc = self._documents()[0]

        # a book without iiif settings must not claim another manuscript's images
        meta = DatabaseBookMeta.load(self.book)
        self.assertEqual(meta.iiifImageApi, '')
        payload = monodi_json_of_document(self.book, doc, 'editor')['document']
        self.assertEqual(payload['source_id'], meta.name)
        self.assertEqual(payload['additionalData']['iiifs'], [])
        self.assertEqual(payload['additionalData']['Melodie_Quelle'], [])

        meta.monodiSourceId = 'Demo Source'
        meta.iiifImageApi = 'https://example.org/iiif/3/'
        meta.iiifSource = 'demo_ms'
        meta.iiifSuffix = '.png'
        meta.to_file(self.book)
        payload = monodi_json_of_document(self.book, doc, 'editor')['document']
        self.assertEqual(payload['source_id'], 'Demo Source')
        self.assertEqual(payload['additionalData']['iiifs'],
                         ['https://example.org/iiif/3/demo_ms%2F' + name + '.png'
                          for name in doc.pages_names])

    def test_download_endpoint(self):
        import io
        import zipfile
        from django.test import Client
        from database.database_permissions import DatabaseBookPermissionFlag
        # the endpoint requires READ; grant it to anonymous like a publicly readable book
        permissions = self.book.get_permissions()
        permissions.permissions.default.set(DatabaseBookPermissionFlag.READ)
        permissions.write()

        doc = self._documents()[0]
        client = Client()
        base = '/api/book/demo/document/'

        response = client.get(base + doc.doc_id + '/download/monodiplus.json')
        self.assertEqual(response.status_code, 200)
        with_meta = json.loads(response.content)
        self.assertEqual(set(with_meta.keys()), {'document', 'notes'})

        # same content without the metadata envelope, for workflows that want the notes alone
        response = client.get(base + doc.doc_id + '/download/monodiplus_notes.json')
        self.assertEqual(response.status_code, 200)
        notes_only = json.loads(response.content)
        self.assertNotIn('document', notes_only)

        def without_uuids(node):
            # every element is given a fresh uuid on serialisation, so two exports of the
            # same chant are equal only once those are dropped
            if isinstance(node, dict):
                return {k: without_uuids(v) for k, v in node.items() if k != 'uuid'}
            if isinstance(node, list):
                return [without_uuids(v) for v in node]
            return node

        self.assertEqual(without_uuids(notes_only), without_uuids(with_meta['notes']))

        response = client.get(base + doc.doc_id + '/download/mei4.zip')
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            self.assertEqual(zf.namelist(), [name + '.xml' for name in doc.pages_names])

        self.assertEqual(client.get(base + doc.doc_id + '/download/nonsense.json').status_code, 400)

        # LocaleMiddleware rewrites every 404 into a redirect to the language-prefixed SPA
        # (the project mounts webapp under i18n_patterns), so drop it to see the view's status
        from django.test import override_settings
        without_locale = [m for m in settings.MIDDLEWARE if 'LocaleMiddleware' not in m]
        with override_settings(MIDDLEWARE=without_locale):
            response = Client().get(base + 'ffffffff-0000-0000-0000-000000000000/download/monodiplus.json')
        self.assertEqual(response.status_code, 404)


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
