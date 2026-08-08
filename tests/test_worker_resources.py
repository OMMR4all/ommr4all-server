import logging
import os
import sys
import time
import unittest
import uuid
from typing import List

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

import ommr4all.settings as settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the test storage; the env var makes spawned task worker processes
# (which re-import settings) use it as well
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from django.contrib.auth.models import Permission, User
from django.test import TestCase
from rest_framework.test import APIClient

from omr.steps.algorithmtypes import AlgorithmTypes, WorkerResource
from omr.steps.step import Step
from restapi.operationworker.operationworker import OperationWorker, Resources
from restapi.operationworker.task import TaskStatusCodes
from restapi.operationworker.taskresources import TaskResource
from restapi.operationworker.taskrunners.taskrunner import TaskRunner
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.operationworker.workerresources import \
    groups_for, resolve_worker_resource, n_workers, InvalidWorkerResourceException, \
    InvalidTrainerParamsException, TRAIN_OPERATIONS, default_n_epoch, resolve_n_epoch


class SleepyTaskRunner(TaskRunner):
    def __init__(self, task_group: List[TaskWorkerGroup], time_s: float):
        super().__init__(None, None, task_group)
        self.id = uuid.uuid4()
        self.time_s = time_s

    def algorithm_meta(self):
        return None

    def identifier(self):
        return self.id

    def run(self, task, com_queue) -> dict:
        time.sleep(self.time_s)
        return {}


class TestMetaPolicy(unittest.TestCase):
    def test_base_defaults(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        self.assertEqual(meta.default_predictor_resource(), WorkerResource.CPU)
        self.assertEqual(meta.allowed_predictor_resources(), [WorkerResource.CPU, WorkerResource.GPU])
        self.assertEqual(meta.default_trainer_resource(), WorkerResource.GPU)
        self.assertEqual(meta.allowed_trainer_resources(), [WorkerResource.CPU, WorkerResource.GPU])

    def test_llm_gpu_only(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_LLM)
        self.assertEqual(meta.default_predictor_resource(), WorkerResource.GPU)
        self.assertEqual(meta.allowed_predictor_resources(), [WorkerResource.GPU])


class TestGroupsFor(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(groups_for(WorkerResource.CPU, training=False), [TaskWorkerGroup.NORMAL_TASKS_CPU])
        self.assertEqual(groups_for(WorkerResource.GPU, training=False), [TaskWorkerGroup.LONG_TASKS_GPU])
        self.assertEqual(groups_for(WorkerResource.CPU, training=True), [TaskWorkerGroup.LONG_TASKS_CPU])
        self.assertEqual(groups_for(WorkerResource.GPU, training=True), [TaskWorkerGroup.LONG_TASKS_GPU])


def _worker_with(groups: List[TaskWorkerGroup]) -> OperationWorker:
    return OperationWorker(resources=Resources([TaskResource(g) for g in groups]), watcher_interval=-1)


class TestResolveWorkerResource(unittest.TestCase):
    ALL_GROUPS = [TaskWorkerGroup.LONG_TASKS_GPU, TaskWorkerGroup.LONG_TASKS_CPU, TaskWorkerGroup.NORMAL_TASKS_CPU]

    def test_explicit_invalid_value(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        with self.assertRaises(InvalidWorkerResourceException):
            resolve_worker_resource(meta, 'tpu', training=False, worker=_worker_with(self.ALL_GROUPS))

    def test_explicit_disallowed(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_LLM)
        with self.assertRaises(InvalidWorkerResourceException):
            resolve_worker_resource(meta, 'cpu', training=False, worker=_worker_with(self.ALL_GROUPS))

    def test_explicit_allowed(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        worker = _worker_with(self.ALL_GROUPS)
        self.assertEqual(resolve_worker_resource(meta, 'gpu', training=False, worker=worker), WorkerResource.GPU)
        self.assertEqual(resolve_worker_resource(meta, 'cpu', training=False, worker=worker), WorkerResource.CPU)

    def test_absent_uses_default(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        worker = _worker_with(self.ALL_GROUPS)
        self.assertEqual(resolve_worker_resource(meta, None, training=False, worker=worker), WorkerResource.CPU)
        self.assertEqual(resolve_worker_resource(meta, None, training=True, worker=worker), WorkerResource.GPU)

    def test_absent_degrades_to_available(self):
        # training defaults to GPU, but on a GPU-less server it degrades to CPU
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        worker = _worker_with([TaskWorkerGroup.LONG_TASKS_CPU, TaskWorkerGroup.NORMAL_TASKS_CPU])
        self.assertEqual(resolve_worker_resource(meta, None, training=True, worker=worker), WorkerResource.CPU)

    def test_absent_gpu_only_never_degrades(self):
        # text_llm allows GPU only: without GPU workers the default stays GPU
        # (the PUT view rejects explicit requests; default resolution must not
        # silently pick a disallowed resource)
        meta = Step.create_meta(AlgorithmTypes.OCR_LLM)
        worker = _worker_with([TaskWorkerGroup.LONG_TASKS_CPU, TaskWorkerGroup.NORMAL_TASKS_CPU])
        self.assertEqual(resolve_worker_resource(meta, None, training=False, worker=worker), WorkerResource.GPU)

    def test_explicit_never_degrades(self):
        meta = Step.create_meta(AlgorithmTypes.OCR_GUPPY)
        worker = _worker_with([TaskWorkerGroup.NORMAL_TASKS_CPU])
        self.assertEqual(resolve_worker_resource(meta, 'gpu', training=False, worker=worker), WorkerResource.GPU)

    def test_n_workers(self):
        worker = _worker_with(self.ALL_GROUPS)
        self.assertEqual(n_workers([TaskWorkerGroup.LONG_TASKS_GPU], worker), 1)
        self.assertEqual(n_workers([TaskWorkerGroup.SHORT_TASKS_CPU], worker), 0)


class TestQueuePosition(unittest.TestCase):
    def test_queue_position(self):
        user = None
        worker = OperationWorker(resources=Resources([
            TaskResource(TaskWorkerGroup.LONG_TASKS_GPU),
            TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU),
        ]), watcher_interval=1)

        running_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 8), user)
        time.sleep(0.5)
        self.assertEqual(worker.status(running_id).code, TaskStatusCodes.RUNNING)
        self.assertEqual(worker.status(running_id).queue_position, -1)

        first_queued = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 1), user)
        second_queued = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 1), user)
        # a CPU task queued behind the GPU tasks competes for other resources:
        # it is picked up immediately and must not count towards their position
        cpu_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 8), user)
        time.sleep(0.5)

        self.assertEqual(worker.status(cpu_id).code, TaskStatusCodes.RUNNING)
        self.assertEqual(worker.status(first_queued).code, TaskStatusCodes.QUEUED)
        self.assertEqual(worker.status(first_queued).queue_position, 0)
        self.assertEqual(worker.status(second_queued).code, TaskStatusCodes.QUEUED)
        self.assertEqual(worker.status(second_queued).queue_position, 1)

        worker.stop(running_id)
        worker.stop(cpu_id)
        time.sleep(3)


class TestWorkerResourcesEndpoint(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('worker_resources_user', password='pw')
        self.client = APIClient()

    def _login(self, user=None):
        self.client.force_authenticate(user=user if user is not None else self.user)

    def test_requires_authentication(self):
        # the worker layout and queue occupancy of the server are not public information
        response = self.client.get('/api/operation/text_llm/worker_resources')
        self.assertEqual(response.status_code, 401)

    def test_text_llm(self):
        self._login()
        response = self.client.get('/api/operation/text_llm/worker_resources')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['operation'], 'text_llm')
        self.assertFalse(data['resources']['cpu']['allowed'])
        self.assertTrue(data['resources']['gpu']['allowed'])
        for info in data['resources'].values():
            for key in ('allowed', 'default', 'n_workers', 'n_free', 'n_tasks_queued'):
                self.assertIn(key, info)

    def test_training_operation(self):
        self._login()
        response = self.client.get('/api/operation/train_symbols/worker_resources')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['resources']['gpu']['allowed'])
        self.assertTrue(data['resources']['cpu']['allowed'])

    def test_unknown_operation(self):
        self._login()
        response = self.client.get('/api/operation/does_not_exist/worker_resources')
        self.assertEqual(response.status_code, 400)


class TestTrainParamsEndpoint(TestCase):
    """The epoch limit offered to the client; the same limit is applied again when training starts."""

    def setUp(self):
        self.user = User.objects.create_user('train_params_user', password='pw')
        self.client = APIClient()

    def test_requires_authentication(self):
        response = self.client.get('/api/operation/train_symbols/train_params')
        self.assertEqual(response.status_code, 401)

    def test_default_user_is_capped_at_the_default(self):
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/operation/train_symbols/train_params')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data['n_epoch_default'], 0)
        self.assertEqual(data['n_epoch_max'], data['n_epoch_default'])

    def test_admin_has_no_limit(self):
        self.user.is_staff = True
        self.user.save()
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/operation/train_symbols/train_params')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['n_epoch_max'])

    def test_granted_permission_lifts_the_limit(self):
        self.user.user_permissions.add(Permission.objects.get(codename='set_training_epochs'))
        self.client.force_authenticate(user=User.objects.get(pk=self.user.pk))
        response = self.client.get('/api/operation/train_symbols/train_params')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['n_epoch_max'])

    def test_unknown_operation(self):
        self.client.force_authenticate(user=self.user)
        # a prediction operation has no trainer settings
        response = self.client.get('/api/operation/text_llm/train_params')
        self.assertEqual(response.status_code, 400)


class TestResolveNEpoch(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('n_epoch_user', password='pw')
        self.algorithm = TRAIN_OPERATIONS['train_symbols']
        self.default = default_n_epoch(self.algorithm)

    def test_unset_keeps_the_algorithm_default(self):
        self.assertIsNone(resolve_n_epoch(self.user, self.algorithm, None))

    def test_lowering_is_allowed_for_everyone(self):
        self.assertEqual(resolve_n_epoch(self.user, self.algorithm, 1), 1)
        self.assertEqual(resolve_n_epoch(self.user, self.algorithm, self.default), self.default)

    def test_raising_is_capped_for_a_regular_user(self):
        self.assertEqual(resolve_n_epoch(self.user, self.algorithm, self.default + 500), self.default)

    def test_raising_is_allowed_for_an_admin(self):
        self.user.is_superuser = True
        self.assertEqual(resolve_n_epoch(self.user, self.algorithm, self.default + 500), self.default + 500)

    def test_rejects_values_below_one(self):
        # mix_default() only replaces None/negative values, so 0 would survive into the trainer
        for value in (0, -1):
            with self.assertRaises(InvalidTrainerParamsException):
                resolve_n_epoch(self.user, self.algorithm, value)

    def test_rejects_garbage(self):
        with self.assertRaises(InvalidTrainerParamsException):
            resolve_n_epoch(self.user, self.algorithm, 'many')


if __name__ == '__main__':
    unittest.main()
