import json
import logging
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from django.contrib.auth.models import Permission, User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from restapi import systeminfo

NVIDIA_SMI_OUTPUT = (
    "0, NVIDIA RTX A5000, 580.65.06, 74, 18200, 24564, 68, 180.50, 230.00\n"
    "1, NVIDIA RTX A5000, 580.65.06, 0, 4, 24564, 31, [N/A], 230.00\n"
)


class SystemInfoTests(unittest.TestCase):
    def setUp(self):
        systeminfo._gpu_cache = None

    def tearDown(self):
        systeminfo._gpu_cache = None

    def test_cpu_and_memory(self):
        info = systeminfo.cpu_and_memory()
        self.assertGreater(info['memory']['total'], 0)
        self.assertGreaterEqual(info['cpu']['count'], 1)
        self.assertEqual(len(info['cpu']['per_cpu']), info['cpu']['count'])

    def test_disk_usage(self):
        disks = systeminfo.disk_usage()
        self.assertGreaterEqual(len(disks), 1)
        for disk in disks:
            self.assertGreater(disk['total'], 0)
            # server-side paths must not leak into the API
            self.assertNotIn('path', disk)

    def test_gpu_stats_parses_nvidia_smi(self):
        completed = subprocess.CompletedProcess([], 0, stdout=NVIDIA_SMI_OUTPUT, stderr='')
        with mock.patch('subprocess.run', return_value=completed):
            gpus, error = systeminfo.gpu_stats(use_cache=False)
        self.assertIsNone(error)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0], {'index': 0, 'name': 'NVIDIA RTX A5000', 'driver_version': '580.65.06',
                                   'utilization': 74.0, 'memory_used': 18200.0, 'memory_total': 24564.0,
                                   'temperature': 68.0, 'power_draw': 180.5, 'power_limit': 230.0})
        # unsupported fields must not break the whole row
        self.assertIsNone(gpus[1]['power_draw'])

    def test_gpu_stats_without_nvidia_smi(self):
        with mock.patch('subprocess.run', side_effect=FileNotFoundError()):
            gpus, error = systeminfo.gpu_stats(use_cache=False)
        self.assertEqual(gpus, [])
        self.assertIsNotNone(error)

    def test_gpu_stats_on_error_exit(self):
        completed = subprocess.CompletedProcess([], 9, stdout='', stderr='Failed to initialize NVML\n')
        with mock.patch('subprocess.run', return_value=completed):
            gpus, error = systeminfo.gpu_stats(use_cache=False)
        self.assertEqual(gpus, [])
        self.assertEqual(error, 'Failed to initialize NVML')


class CudaStatusTests(unittest.TestCase):
    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    @staticmethod
    def _reset():
        systeminfo._cuda_result = None
        systeminfo._cuda_running = False

    def _await_ready(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = systeminfo.cuda_status()
            if status.get('state') == 'ready':
                return status
            time.sleep(0.02)
        self.fail('the CUDA probe did not finish')

    def test_probes_in_the_background(self):
        probe = json.dumps({'torch_version': '2.11.0+cu130', 'cuda_built': '13.0',
                            'arch_list': ['sm_61', 'sm_86'], 'available': True,
                            'devices': [{'index': 0, 'name': 'RTX A5000', 'capability': '8.6',
                                         'sm': 'sm_86', 'supported': True, 'compute_ok': True}]})
        completed = subprocess.CompletedProcess([], 0, stdout=probe + '\n', stderr='')
        with mock.patch('subprocess.run', return_value=completed):
            # the first call must not block on importing torch
            self.assertEqual(systeminfo.cuda_status(), {'state': 'checking'})
            status = self._await_ready()
        self.assertTrue(status['available'])
        self.assertTrue(status['devices'][0]['compute_ok'])

    def test_reports_a_build_without_kernels_for_the_card(self):
        # the Pascal case: torch sees the GPU but cannot launch a kernel on it
        probe = json.dumps({'torch_version': '2.11.0+cu130', 'cuda_built': '13.0',
                            'arch_list': ['sm_75', 'sm_86'], 'available': True,
                            'devices': [{'index': 0, 'name': 'GTX 1080', 'capability': '6.1',
                                         'sm': 'sm_61', 'supported': False, 'compute_ok': False,
                                         'error': 'no kernel image is available for execution on the device'}]})
        completed = subprocess.CompletedProcess([], 0, stdout=probe, stderr='')
        with mock.patch('subprocess.run', return_value=completed):
            systeminfo.cuda_status()
            status = self._await_ready()
        self.assertTrue(status['available'])
        self.assertFalse(status['devices'][0]['supported'])
        self.assertFalse(status['devices'][0]['compute_ok'])

    def test_unreadable_probe_output(self):
        completed = subprocess.CompletedProcess([], 0, stdout='segmentation fault', stderr='')
        with mock.patch('subprocess.run', return_value=completed):
            systeminfo.cuda_status()
            status = self._await_ready()
        self.assertIn('error', status)

    def test_result_is_cached_until_refreshed(self):
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({'available': False}), stderr='')
        with mock.patch('subprocess.run', return_value=completed) as run:
            systeminfo.cuda_status()
            self._await_ready()
            self.assertEqual(run.call_count, 1)
            systeminfo.cuda_status()
            systeminfo.cuda_status()
            self.assertEqual(run.call_count, 1)
            systeminfo.cuda_status(refresh=True)
            self._await_ready()
            self.assertEqual(run.call_count, 2)


class SystemResourcesViewTests(APITestCase):
    def setUp(self):
        systeminfo._gpu_cache = None
        systeminfo._cuda_result = {'available': False}     # skip the torch probe
        self.user = User.objects.create_user('resources_user', password='pw')

    def tearDown(self):
        systeminfo._gpu_cache = None
        systeminfo._cuda_result = None

    def _login(self):
        response = self.client.post(reverse('token_obtain_pair'),
                                    {'username': 'resources_user', 'password': 'pw'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + response.data['access'])

    def test_requires_tasks_list_permission(self):
        self._login()
        response = self.client.get('/api/system_resources', format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)

    def test_reports_resources(self):
        self.user.user_permissions.add(Permission.objects.get(codename='tasks_list'))
        self._login()
        with mock.patch('subprocess.run', side_effect=FileNotFoundError()):
            response = self.client.get('/api/system_resources', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        body = response.json()
        self.assertGreater(body['memory']['total'], 0)
        # a host without nvidia-smi must still render: no gpus, but a reason
        self.assertEqual(body['gpus'], [])
        self.assertIsNotNone(body['gpu_error'])
        self.assertEqual(body['cuda']['state'], 'ready')
        self.assertGreater(len(body['workers']), 0)
        for worker in body['workers']:
            self.assertIn('group', worker)
            self.assertIn('used', worker)
        self.assertEqual(body['queue']['n_running'], 0)


if __name__ == '__main__':
    unittest.main()
