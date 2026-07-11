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

from django.test import Client

from omr.steps.algorithmtypes import AlgorithmTypes, WorkerResource
from omr.steps.step import Step
from restapi.operationworker.operationworker import OperationWorker, Resources
from restapi.operationworker.task import TaskStatusCodes
from restapi.operationworker.taskresources import TaskResource
from restapi.operationworker.taskrunners.taskrunner import TaskRunner
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.operationworker.workerresources import \
    groups_for, resolve_worker_resource, n_workers, InvalidWorkerResourceException


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


class TestWorkerResourcesEndpoint(unittest.TestCase):
    def test_text_llm(self):
        response = Client().get('/api/operation/text_llm/worker_resources')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['operation'], 'text_llm')
        self.assertFalse(data['resources']['cpu']['allowed'])
        self.assertTrue(data['resources']['gpu']['allowed'])
        for info in data['resources'].values():
            for key in ('allowed', 'default', 'n_workers', 'n_free', 'n_tasks_queued'):
                self.assertIn(key, info)

    def test_training_operation(self):
        response = Client().get('/api/operation/train_symbols/worker_resources')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['resources']['gpu']['allowed'])
        self.assertTrue(data['resources']['cpu']['allowed'])

    def test_unknown_operation(self):
        response = Client().get('/api/operation/does_not_exist/worker_resources')
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
