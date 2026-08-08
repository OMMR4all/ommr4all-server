import logging
import os
import shutil
import sys

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

from omr.steps.algorithmtypes import AlgorithmGroups, AlgorithmTypes
from omr.steps.step import Step
from restapi.views.administrativedefaultmodels import default_model_slots


class TestDefaultModelSlots(DjangoTestCase):
    """The set of algorithms a default model can be configured for."""

    def setUp(self):
        self.user = User.objects.create_user('default_models_user', password='pw')
        self.admin = User.objects.create_superuser('default_models_admin', password='pw')

    def _client(self, user):
        client = APIClient()
        client.force_authenticate(user=user)
        return client

    def test_slots_are_registered_model_based_algorithms(self):
        for t, aliases in default_model_slots():
            self.assertTrue(t.uses_model(), t)
            self.assertIn(t, Step.METAS, "{} is not registered in omr/steps/step.py".format(t))
            for alias in aliases:
                # an alias is configured by this slot because it reads the same directory
                self.assertEqual(Step.meta(alias).model_dir(), Step.meta(t).model_dir())

    def test_no_two_slots_share_a_model_directory(self):
        # two slots writing the same directory would silently overwrite each other
        dirs = [Step.meta(t).model_dir() for t, _ in default_model_slots()]
        self.assertEqual(len(dirs), len(set(dirs)), dirs)

    def test_aliased_types_collapse_into_one_slot(self):
        slots = {t: aliases for t, aliases in default_model_slots()}
        self.assertNotIn(AlgorithmTypes.LAYOUT_SIMPLE_LYRICS, slots)
        self.assertIn(AlgorithmTypes.LAYOUT_SIMPLE_LYRICS,
                      slots[AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO])

    def test_syllables_are_configured_separately_from_text(self):
        # the step runs a text model but must be able to use a different one than the text step
        slots = {t: aliases for t, aliases in default_model_slots()}
        self.assertIn(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH, slots)
        self.assertEqual(slots[AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH], [])
        self.assertNotIn(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH, slots[AlgorithmTypes.OCR_GUPPY])
        self.assertEqual(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH.group(), AlgorithmGroups.SYLLABLES)

    def test_the_syllable_step_falls_back_to_the_text_default(self):
        # no default is shipped for the syllable step; without an own one it uses the text model
        model = Step.meta(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH).default_model_for_style('french14')
        self.assertTrue(model.path.endswith(os.path.join('french14', 'text_guppy')), model.path)

    def test_the_syllable_step_never_predicts_with_an_empty_default_directory(self):
        # internal_storage ships a bare meta.json for every algorithm; handing that to the
        # predictor fails the prediction, so a directory without weights must not be used
        from database import DatabaseBook
        model = Step.meta(AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH).selected_model_for_book(
            DatabaseBook('demo'))
        self.assertTrue(model.has_weights(), model.path)

    def test_the_syllable_default_is_stored_next_to_the_text_default(self):
        from database.database_available_models import DatabaseAvailableModels
        source = 'i/french14/text_guppy/text_guppy'
        style_dir = DatabaseAvailableModels.local_default_model_path_for_style('teststyle')
        target = os.path.join(style_dir, AlgorithmTypes.SYLLABLES_FROM_TEXT_TORCH.value)
        text_target = os.path.join(style_dir, 'text_guppy')
        self.assertFalse(os.path.exists(target))
        try:
            response = self._client(self.admin).put(
                '/api/administrative/default_models/type/syllables_from_text_torch/style/teststyle',
                {'id': source}, format='json')
            self.assertEqual(response.status_code, 200, response.content)
            self.assertTrue(os.path.exists(os.path.join(target, 'meta.json')))
            self.assertFalse(os.path.exists(text_target), 'the text default must stay untouched')
        finally:
            shutil.rmtree(target, ignore_errors=True)

    def test_slots_endpoint(self):
        body = self._client(self.user).get('/api/administrative/default_models/slots').json()
        self.assertEqual([s['type'] for s in body['slots']],
                         [t.value for t, _ in default_model_slots()])
        for slot in body['slots']:
            AlgorithmGroups(slot['group'])  # must not raise

    def test_get_of_a_type(self):
        response = self._client(self.user).get(
            '/api/administrative/default_models/type/staff_lines_pc_torch/style/french14')
        self.assertEqual(response.status_code, 200, response.content)
        self.assertIn('has_own_default', response.json())

    def test_get_of_a_type_without_a_default_model(self):
        # end2end has no french14 fallback, the response must still be well formed
        response = self._client(self.user).get(
            '/api/administrative/default_models/type/end2end_swin/style/french14')
        self.assertEqual(response.status_code, 200, response.content)
        self.assertFalse(response.json()['has_own_default'])

    def test_unknown_or_model_less_types_are_rejected(self):
        client = self._client(self.admin)
        for t in ['not_an_algorithm', 'preprocessing', 'syllables_in_order']:
            response = client.get('/api/administrative/default_models/type/{}/style/french14'.format(t))
            self.assertEqual(response.status_code, 400, (t, response.content))

    def test_the_group_route_uses_the_primary_algorithm_of_the_group(self):
        client = self._client(self.user)
        by_group = client.get('/api/administrative/default_models/group/stafflines/style/french14')
        by_type = client.get('/api/administrative/default_models/type/{}/style/french14'.format(
            AlgorithmGroups.STAFF_LINES.types()[0].value))
        self.assertEqual(by_group.status_code, 200, by_group.content)
        self.assertEqual(by_group.json(), by_type.json())

    def test_a_group_the_server_has_no_algorithms_for_is_rejected(self):
        # the client's group enum contains 'documents' and 'search', the server's does not
        response = self._client(self.admin).get(
            '/api/administrative/default_models/group/documents/style/french14')
        self.assertEqual(response.status_code, 400, response.content)

    def test_writing_a_default_requires_the_permission(self):
        response = self._client(self.user).put(
            '/api/administrative/default_models/type/staff_lines_pc_torch/style/french14',
            {'id': 'i/french14/staff_lines_pc_torch/staff_lines_pc_torch'}, format='json')
        self.assertEqual(response.status_code, 401, response.content)
