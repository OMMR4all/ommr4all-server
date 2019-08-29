import logging
import sys
import unittest
import time
from typing import List
import uuid

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

from restapi.operationworker.taskresources import TaskResource
from restapi.operationworker.operationworker import OperationWorker, Resources
from restapi.operationworker.task import TaskStatusCodes, TaskNotFoundException
from restapi.operationworker.taskrunners.taskrunner import TaskRunner
from restapi.operationworker.taskworkergroup import TaskWorkerGroup


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


class TestSkeduler(unittest.TestCase):
    def test_skeduler(self):
        user = None
        default_resources: Resources = Resources([
            TaskResource(g) for g in (
                    [TaskWorkerGroup.LONG_TASKS_GPU] * 3 +
                    [TaskWorkerGroup.LONG_TASKS_CPU] * 2 +
                    [TaskWorkerGroup.NORMAL_TASKS_CPU] * 5 +
                    [TaskWorkerGroup.SHORT_TASKS_CPU] * 3)
        ])
        worker = OperationWorker(resources=default_resources, watcher_interval=1)

        full_gpu_tasks = [SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 8) for i in range(3)]
        full_gpu_task_ids = [worker.put(task, user) for task in full_gpu_tasks]

        time.sleep(0.5)
        self.assertEqual(3, worker.resources.n_used())
        
        # all gpu tasks must run
        for task in full_gpu_task_ids:
            self.assertEqual(worker.status(task).code, TaskStatusCodes.RUNNING)

        # add two other ones, which must be queued
        queued_task = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 2), user)
        time.sleep(0.5)
        self.assertEqual(3, worker.resources.n_used())
        self.assertEqual(worker.status(queued_task).code, TaskStatusCodes.QUEUED)

        # cancel one job, then the queued task must run
        worker.stop(full_gpu_task_ids[0])
        time.sleep(0.5)
        self.assertEqual(3, worker.resources.n_used())
        self.assertEqual(worker.status(queued_task).code, TaskStatusCodes.RUNNING)
        with self.assertRaises(TaskNotFoundException):
            worker.status(full_gpu_task_ids[0])

        # add a job that can also run on CPU, this must be skeduled aswell
        job_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU, TaskWorkerGroup.LONG_TASKS_CPU], 8), user)
        time.sleep(0.5)
        self.assertEqual(4, worker.resources.n_used())
        self.assertEqual(worker.status(job_id).code, TaskStatusCodes.RUNNING)

        # add a new queued job to GPU, this must be queued until old queued job stops
        job_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 8), user)
        time.sleep(0.5)
        self.assertEqual(4, worker.resources.n_used())
        self.assertEqual(worker.status(job_id).code, TaskStatusCodes.QUEUED)
        # wait 2 secs for the first job to stop, then the job must run
        time.sleep(2)
        self.assertEqual(4, worker.resources.n_used())
        self.assertEqual(worker.status(job_id).code, TaskStatusCodes.RUNNING)
        # and the other job must be marked as finished
        self.assertEqual(worker.status(queued_task).code, TaskStatusCodes.FINISHED)
        
        # wait for all jobs to stop
        time.sleep(10)
        self.assertEqual(0, worker.resources.n_used())


if __name__ == '__main__':
    unittest.main()
