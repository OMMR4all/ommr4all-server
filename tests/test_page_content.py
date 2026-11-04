"""The index-backed structural counts the batch predicates run on.

`PageContent` replaces a full PcGts parse per page in `unprocessed`/`runnable`, so the
counts must mean exactly what the parsed page means -- and a whole-book page selection
must not fall back to parsing after all.
"""
import json
import logging
import os
import sys
from unittest import mock

import ommr4all.settings as settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django  # noqa: E402
django.setup()

from django.contrib.auth.models import User  # noqa: E402
from django.urls import reverse  # noqa: E402
from rest_framework import status  # noqa: E402
from rest_framework.test import APITestCase  # noqa: E402

from database import DatabaseBook, book_index  # noqa: E402
from database.database_page import DatabasePage  # noqa: E402
from database.file_formats.pcgts.page.definitions import BlockType  # noqa: E402
from database.models.book_index import PageIndex  # noqa: E402
from database.page_content import PageContent  # noqa: E402

# the steps whose predicates read the page content; one per distinct predicate
CONTENT_OPERATIONS = ['staff_lines_pc_torch', 'staff_lines_correction', 'layout_simple_lyrics',
                      'symbols_pc_torch', 'text_guppy', 'text_llm', 'syllables_in_order',
                      'end2end_convnext', 'symbols_pattern_matcher']


class PageContentTestCase(APITestCase):
    def setUp(self):
        self.book = DatabaseBook('demo')
        self.pages = self.book.pages()

    def test_the_json_counts_match_the_parsed_page(self):
        """from_page is the definition; from_pcgts_json is what the index stores."""
        for page in self.pages:
            with open(page.local_file_path('pcgts.json')) as f:
                raw = json.load(f)
            self.assertEqual(PageContent.from_pcgts_json(raw),
                             PageContent.from_page(page.pcgts().page), page.page)

    def test_the_counts_answer_the_predicates_like_the_parsed_page(self):
        """Each assertion is the formulation the predicate used before it moved to the index."""
        for page in self.pages:
            p = page.pcgts().page
            content = PageContent.from_page(p)
            name = page.page
            self.assertEqual(content.n_music_blocks, len(p.music_blocks()), name)
            self.assertEqual(content.n_text_blocks, len(p.text_blocks()), name)
            self.assertEqual(content.n_blocks(BlockType.LYRICS),
                             len(p.blocks_of_type([BlockType.LYRICS])), name)
            self.assertEqual(content.has_music_symbols,
                             not all(len(l.symbols) == 0 for l in p.all_music_lines()), name)
            self.assertEqual(content.has_staff_lines,
                             any(len(l.staff_lines) > 0 for l in p.all_music_lines()), name)
            self.assertEqual(content.connections, len(p.annotations.connections), name)
            llm_types = [BlockType.LYRICS, BlockType.PARAGRAPH, BlockType.HEADING]
            self.assertEqual(content.n_lines_without_syllables(llm_types) > 0,
                             any(len(l.sentence.syllables) == 0
                                 for l in p.all_lines_by_type(llm_types)), name)

    def test_syllables_without_text_and_dangling_connections_do_not_count(self):
        """What the pcgts loader drops (Sentence.from_json, Connection.from_json) must not
        show up in the counts either -- otherwise a page would look processed to the
        predicates while the parsed page says it is not."""
        raw = {'page': {'blocks': [
            {'id': 'm', 'type': 'music', 'lines': [{'id': 'ml', 'symbols': [], 'staffLines': []}]},
            {'id': 't', 'type': 'lyrics', 'lines': [
                {'id': 'tl', 'sentence': {'syllables': [{'id': 's', 'text': ''}]}}]},
        ], 'annotations': {'connections': [
            {'musicID': 'm', 'textID': 'gone', 'syllableConnectors': []}]}}}
        content = PageContent.from_pcgts_json(raw)
        self.assertEqual(content.n_lines_without_syllables(BlockType.LYRICS), 1)
        self.assertEqual(content.connections, 0)
        self.assertFalse(content.has_music_symbols)
        self.assertFalse(content.has_staff_lines)

    def test_a_page_without_a_pcgts_file_is_empty(self):
        content = PageContent.from_pcgts_json(None)
        self.assertEqual(content, PageContent())
        self.assertEqual(content.n_music_blocks, 0)


class PageContentIndexTestCase(APITestCase):
    def setUp(self):
        url = reverse('token_obtain_pair')
        User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        response = self.client.post(url, {'username': 'user', 'password': 'user'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer {0}'.format(response.data['access']))
        self.book = DatabaseBook('demo')

    def _page_selection(self, count):
        """The workflow tab's request: one selection evaluated against all enabled steps."""
        return self.client.post(
            '/api/book/demo/operation/text_llm/page_selection',
            {'count': count, 'pages': [], 'selected_pages_range_as_regex': '',
             'operations': CONTENT_OPERATIONS}, format='json')

    def test_a_whole_book_selection_parses_no_pcgts(self):
        """The regression this exists for: parsing one pcgts per page took 47 s on a 532
        page book, long enough for the server to kill the request the workflow tab had
        only asked a page count from."""
        book_index.index_book(self.book)
        for count in ('all', 'unprocessed'):
            with mock.patch.object(DatabasePage, 'pcgts', autospec=True,
                                   side_effect=AssertionError('parsed a pcgts')), \
                    mock.patch.object(DatabasePage, 'pcgts_cached', autospec=True,
                                      side_effect=AssertionError('parsed a pcgts')):
                response = self._page_selection(count)
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_the_selection_is_the_same_with_and_without_the_stored_counts(self):
        """A book whose rows predate the content column must select exactly the same pages."""
        expected = json.loads(self._page_selection('unprocessed').content)
        PageIndex.objects.filter(book__name='demo').update(content=None)
        self.assertEqual(json.loads(self._page_selection('unprocessed').content), expected)
        # the backfill persisted the counts, so the next request does not recompute them
        self.assertFalse(PageIndex.objects.filter(book__name='demo', content=None).exists())

    def test_an_in_memory_pcgts_wins_over_the_index(self):
        """The editor posts its unsaved pcgts with a single-page operation; the predicates
        must see that state, not the last saved one."""
        page = self.book.page('page_test_symbol_detection_001')
        book_index.index_page(page)
        self.assertTrue(page.page_content().has_music_symbols)

        d = page.pcgts().to_json()
        for block in d['page']['blocks']:
            for line in block.get('lines', []):
                line['symbols'] = []
        stripped = self.book.page(page.page)
        stripped.pcgts_from_dict(d)
        self.assertTrue(stripped.has_page_content())
        self.assertFalse(stripped.page_content().has_music_symbols)

    def test_the_stored_counts_follow_the_pcgts(self):
        page = self.book.page('page_test_symbol_detection_001')
        row = book_index.index_page(page)
        self.assertEqual(PageContent.from_dict(row.content),
                         PageContent.from_page(page.pcgts().page))
        # a changed pcgts invalidates the row by its mtime, like every other indexed field
        PageIndex.objects.filter(book__name='demo', name=page.page).update(
            content=PageContent().to_dict(), pcgts_mtime=0)
        self.assertEqual(book_index.page_content(self.book.page(page.page)),
                         PageContent.from_page(page.pcgts().page))
