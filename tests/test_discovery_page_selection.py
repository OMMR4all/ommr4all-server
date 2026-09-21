import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import ommr4all.settings as settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')
import django; django.setup()

from database import DatabaseBook
from database.file_formats.performance.pageprogress import Locks, PageProgress
from omr.discovery.config import RunConfig
from omr.discovery.page_selection import SUGGESTIONS_FILE, select_representative_pages
from omr.discovery.runner import run_page_suggestions


class PageCoreSetSelectionTest(unittest.TestCase):
    def test_corrected_pages_seed_diverse_farthest_first_selection(self):
        embeddings = {
            'corrected': np.array([1.0, 0.0]),
            'near': np.array([0.99, 0.1]),
            'different': np.array([0.0, 1.0]),
            'opposite': np.array([-1.0, 0.0]),
        }

        selected = select_representative_pages(
            embeddings, ['near', 'different', 'opposite'], ['corrected'], 2)

        self.assertEqual([item.page for item in selected], ['opposite', 'different'])
        self.assertEqual(selected[0].nearest_reference_page, 'corrected')
        self.assertAlmostEqual(selected[0].novelty_score, 2.0)

    def test_unseeded_selection_starts_at_medoid_not_outlier(self):
        embeddings = {
            'edge': np.array([1.0, 0.0]),
            'medoid': np.array([0.9, 0.2]),
            'other': np.array([0.0, 1.0]),
        }

        selected = select_representative_pages(embeddings, embeddings, [], 2)

        self.assertEqual(selected[0].page, 'medoid')
        self.assertEqual(selected[0].reason, 'representative_seed')
        self.assertEqual(selected[1].page, 'other')

    def test_rejects_empty_requested_batch(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            select_representative_pages({'page': np.ones(2)}, ['page'], [], 0)


class PageSuggestionPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book_name = 'discovery_page_suggestion_test'
        cls.book_path = os.path.join(settings.PRIVATE_MEDIA_ROOT, cls.book_name)
        if os.path.exists(cls.book_path):
            shutil.rmtree(cls.book_path)
        shutil.copytree(DatabaseBook('demo').local_path(), cls.book_path)
        cls.out = tempfile.mkdtemp(prefix='ommr-page-suggestion-test-')
        cls.corrected = 'page_test_symbol_detection_001'
        cls.candidate = 'page_test_symbol_detection_002'
        progress_path = os.path.join(cls.book_path, 'pages', cls.corrected,
                                     'page_progress.json')
        PageProgress(locked={Locks.SYMBOLS: True}).to_json_file(progress_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.book_path, ignore_errors=True)
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_stub_pipeline_uses_symbol_locks_and_writes_ranked_batch(self):
        cfg = RunConfig(book=self.book_name, pages=[self.corrected, self.candidate])
        cfg.features.backend = 'stub'
        cfg.page_suggestions.count = 1

        run_dir = run_page_suggestions(cfg, self.out)

        with open(os.path.join(run_dir, SUGGESTIONS_FILE)) as handle:
            result = json.load(handle)
        self.assertEqual(result['corrected_source'], 'symbols_lock')
        self.assertIn(self.corrected, result['corrected_pages'])
        self.assertIn(self.corrected, result['embedded_corrected_pages'])
        self.assertEqual([item['page'] for item in result['suggestions']], [self.candidate])
        self.assertEqual(result['suggestions'][0]['reason'], 'farthest_from_reference')


if __name__ == '__main__':
    unittest.main()
