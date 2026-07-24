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
    def _status_is(self, worker, task_id, code) -> bool:
        try:
            return worker.status(task_id).code == code
        except TaskNotFoundException:
            return False

    def _wait_until(self, predicate, timeout=30.0, interval=0.05):
        """Poll until predicate() is true, or fail after timeout seconds.

        Task workers are spawned (a fresh interpreter that runs django.setup()
        before executing the task body), so the wall-clock delay between queuing
        a task and it actually running/finishing is dominated by process startup
        and varies widely between machines. Waiting for the expected state keeps
        this test fast on quick machines and robust on slow CI runners, instead
        of relying on fixed sleeps that assume a particular startup speed.
        """
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return
            time.sleep(interval)
        # final evaluation so the caller's assertion reports the real state
        self.assertTrue(predicate())

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

        # all gpu tasks must run (resources are marked used synchronously by the
        # task creator when it schedules a task, independent of worker startup)
        for task in full_gpu_task_ids:
            self._wait_until(lambda t=task: self._status_is(worker, t, TaskStatusCodes.RUNNING))
        self.assertEqual(3, worker.resources.n_used())

        # add another one, which must be queued (no free gpu resource)
        queued_task = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 2), user)
        time.sleep(0.5)
        self.assertEqual(3, worker.resources.n_used())
        self.assertEqual(worker.status(queued_task).code, TaskStatusCodes.QUEUED)

        # cancel one job, then the queued task must run
        worker.stop(full_gpu_task_ids[0])
        self._wait_until(lambda: self._status_is(worker, queued_task, TaskStatusCodes.RUNNING))
        self.assertEqual(3, worker.resources.n_used())
        with self.assertRaises(TaskNotFoundException):
            worker.status(full_gpu_task_ids[0])

        # add a job that can also run on CPU, this must be skeduled aswell
        cpu_job_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU, TaskWorkerGroup.LONG_TASKS_CPU], 8), user)
        self._wait_until(lambda: self._status_is(worker, cpu_job_id, TaskStatusCodes.RUNNING))
        self.assertEqual(4, worker.resources.n_used())

        # add a new queued job to GPU, this must be queued until the 2s job stops
        gpu_job_id = worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 8), user)
        time.sleep(0.5)
        self.assertEqual(4, worker.resources.n_used())
        self.assertEqual(worker.status(gpu_job_id).code, TaskStatusCodes.QUEUED)

        # once the 2s job finishes it frees a gpu, then the queued job must run
        self._wait_until(lambda: self._status_is(worker, gpu_job_id, TaskStatusCodes.RUNNING))
        self.assertEqual(4, worker.resources.n_used())
        # and the other job must be marked as finished (finished strictly before
        # the gpu it freed was handed to gpu_job_id above)
        self.assertEqual(worker.status(queued_task).code, TaskStatusCodes.FINISHED)

        # wait for all remaining jobs to stop
        self._wait_until(lambda: worker.resources.n_used() == 0)


if __name__ == '__main__':
    unittest.main()
