import json
import logging
import os
import shutil
import sys
import tempfile

import ommr4all.settings as settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase as DjangoTestCase
from rest_framework.test import APIClient

from database.model import MetaId, Model, ModelMeta, ModelsId
from omr.steps.algorithmtypes import AlgorithmTypes
from restapi.views import administrativemodels
from restapi.views.administrativemodels import collect_models, parse_model_id


def _write_model(path: str, meta: ModelMeta, weights: bool = True):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, Model.META_FILE), 'w') as f:
        f.write(meta.to_json())
    if weights:
        with open(os.path.join(path, 'best.torch'), 'wb') as f:
            f.write(b'0' * 2048)


class TestModelUsage(DjangoTestCase):
    """The last-used sidecar written on every prediction (see omr/steps/predictorcache.py)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        _write_model(self.dir, ModelMeta(id='test'))
        self.model = administrativemodels.ModelAtPath(self.dir, 'e/book/symbols_pc_torch/2020-01-01T00:00:00')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_unused_model(self):
        usage = self.model.usage()
        self.assertIsNone(usage.last_used)
        self.assertEqual(usage.n_used, 0)

    def test_mark_used_counts_up(self):
        self.model.mark_used()
        self.model.mark_used()
        usage = self.model.usage()
        self.assertEqual(usage.n_used, 2)
        self.assertIsNotNone(usage.last_used)

    def test_mark_used_does_not_touch_the_meta(self):
        # the predictor cache keys on the mtime of meta.json -- writing the usage there would
        # throw away the cached predictor (and reload the weights) on every prediction
        meta_path = os.path.join(self.dir, Model.META_FILE)
        before = os.path.getmtime(meta_path)
        self.model.mark_used()
        self.assertEqual(os.path.getmtime(meta_path), before)

    def test_mark_used_never_raises(self):
        shutil.rmtree(self.dir)
        self.model.mark_used()   # must not raise, a failed prediction would be worse

    def test_copy_records_its_source(self):
        target_dir = tempfile.mkdtemp()
        shutil.rmtree(target_dir)
        try:
            source = Model(MetaId(ModelsId.parse('e/demo/symbols_pc_torch', []), 'src'),
                           meta=ModelMeta(id='e/demo/symbols_pc_torch/src'))
            os.makedirs(source.path, exist_ok=True)
            source.save_meta()
            target = Model(MetaId(ModelsId.from_internal('teststyle', AlgorithmTypes.SYMBOLS_PC_TORCH),
                                  'symbols_pc_torch'))
            source.copy_to(target, override=True)
            with open(target.meta_path) as f:
                stored = json.load(f)
            self.assertEqual(stored['source_id'], source.id())
        finally:
            shutil.rmtree(target.path, ignore_errors=True)
            shutil.rmtree(source.path, ignore_errors=True)
            shutil.rmtree(target_dir, ignore_errors=True)


class TestAdministrativeModels(DjangoTestCase):
    """Listing and pruning of the models of the test storage."""

    BOOK = 'demo'
    DIR = 'symbols_pc_torch'
    OLD = '2020-01-01T00:00:00'
    NEW = '2021-01-01T00:00:00'

    def setUp(self):
        self.user = User.objects.create_user('models_user', password='pw')
        self.admin = User.objects.create_superuser('models_admin', password='pw')
        self.models_root = os.path.join(settings.PRIVATE_MEDIA_ROOT, self.BOOK, 'models', self.DIR)
        _write_model(os.path.join(self.models_root, self.OLD), ModelMeta(id='old'))
        _write_model(os.path.join(self.models_root, self.NEW), ModelMeta(id='new'), weights=False)

    def tearDown(self):
        for name in (self.OLD, self.NEW):
            shutil.rmtree(os.path.join(self.models_root, name), ignore_errors=True)

    def _client(self, user):
        client = APIClient()
        client.force_authenticate(user=user)
        return client

    def _id(self, name):
        return '/'.join(['e', self.BOOK, self.DIR, name])

    def _entry(self, models, name):
        return [m for m in models if m['id'] == self._id(name)][0]

    def test_listing_reports_size_weights_and_usage(self):
        models = collect_models()
        old = self._entry(models, self.OLD)
        self.assertTrue(old['hasMeta'])
        self.assertTrue(old['hasWeights'])
        self.assertGreaterEqual(old['size'], 2048)
        self.assertIsNone(old['lastUsed'])
        # a run that stopped before writing weights is a prune candidate, not a model
        self.assertFalse(self._entry(models, self.NEW)['hasWeights'])

    def test_the_newest_model_of_a_book_and_step_is_protected(self):
        models = collect_models()
        self.assertIn('newest', self._entry(models, self.NEW)['protection'])
        self.assertEqual(self._entry(models, self.OLD)['protection'], [])

    def test_the_default_models_are_protected(self):
        defaults = [m for m in collect_models() if m['storage'] == 'i']
        self.assertTrue(defaults)
        for d in defaults:
            self.assertIn('is_default', d['protection'])

    def test_prune_deletes_only_unprotected_models(self):
        client = self._client(self.admin)
        response = client.post('/api/administrative/models/prune',
                               {'ids': [self._id(self.OLD), self._id(self.NEW)]}, format='json')
        self.assertEqual(response.status_code, 200, response.content)
        body = response.json()
        self.assertEqual(body['deleted'], [self._id(self.OLD)])
        self.assertEqual([r['reason'] for r in body['refused']], ['newest'])
        self.assertGreaterEqual(body['freed'], 2048)
        self.assertFalse(os.path.exists(os.path.join(self.models_root, self.OLD)))
        self.assertTrue(os.path.exists(os.path.join(self.models_root, self.NEW)))

    def test_prune_refuses_unknown_ids(self):
        response = self._client(self.admin).post('/api/administrative/models/prune',
                                                 {'ids': ['e/demo/symbols_pc_torch/does_not_exist']},
                                                 format='json')
        self.assertEqual(response.json()['refused'], [{'id': 'e/demo/symbols_pc_torch/does_not_exist',
                                                       'reason': 'unknown'}])

    def test_ids_outside_the_storage_are_not_addressable(self):
        for bad in ['', 'nonsense', 'x/demo/dir/name', 'e/demo/dir', 'e/../../etc/passwd/x',
                    'e/demo/symbols_pc_torch/../../..']:
            self.assertIsNone(parse_model_id(bad), bad)

    def test_listing_requires_an_administrator(self):
        self.assertEqual(self._client(self.user).get('/api/administrative/models').status_code, 401)
        self.assertEqual(self._client(self.admin).get('/api/administrative/models').status_code, 200)

    def test_prune_requires_an_administrator(self):
        response = self._client(self.user).post('/api/administrative/models/prune',
                                                {'ids': [self._id(self.OLD)]}, format='json')
        self.assertEqual(response.status_code, 401)
        self.assertTrue(os.path.exists(os.path.join(self.models_root, self.OLD)))
