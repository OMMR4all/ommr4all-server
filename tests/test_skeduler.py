import logging
import sys
import unittest
import time
from typing import List
import uuid

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

from unittest import mock

from restapi.operationworker import taskworkerthread
from restapi.operationworker.taskresources import TaskResource
from restapi.operationworker.operationworker import OperationWorker, Resources
from restapi.operationworker.task import TaskStatusCodes, TaskNotFoundException
from restapi.operationworker.taskqueue import TaskQueue
from restapi.operationworker.taskrunners.taskrunner import TaskRunner
from restapi.operationworker.taskwatcher import TaskWatcher
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.operationworker.workerresources import n_free, n_workers


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


class SchedulerTestCase(unittest.TestCase):
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


class TestSkeduler(SchedulerTestCase):
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
        self.addCleanup(worker.shutdown)

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


class _StuckProcess:
    """Stands in for a worker process in uninterruptible sleep (D state).

    It accepts every signal without complaining and never dies -- which is precisely why
    the unbounded ``join()`` it used to be terminated with never returned.
    """

    def __init__(self, gpu_id: int):
        self.name = 'stuck_worker_gpu{}'.format(gpu_id)
        self.pid = -1
        self.joins = 0

    def is_alive(self):
        return True

    def terminate(self):
        pass

    def kill(self):
        pass

    def join(self, timeout=None):
        self.joins += 1
        if timeout:
            time.sleep(timeout)


class _StuckQueue:
    def put(self, item):
        pass

    def close(self):
        pass

    def cancel_join_thread(self):
        pass


class _StuckWorkerProcess(taskworkerthread._WorkerProcess):
    """A _WorkerProcess bound to a process that cannot be killed.

    Deliberately inherits shutdown()/poll_dead(), so the test exercises the real
    escalation logic instead of a reimplementation of it.
    """

    def __init__(self, gpu_id: int, com_queue):
        self.work_queue = _StuckQueue()
        self.process = _StuckProcess(gpu_id)


def _only_gpu_workers_are_stuck():
    """Patch so that GPU slots get an unkillable worker and CPU slots a real one."""
    real = taskworkerthread._WorkerProcess

    def factory(gpu_id, com_queue):
        if gpu_id >= 0:
            return _StuckWorkerProcess(gpu_id, com_queue)
        return real(gpu_id, com_queue)

    return mock.patch.object(taskworkerthread, '_WorkerProcess', new=factory)


class TestUnkillableWorker(SchedulerTestCase):
    """Regression test for the incident of 2026-08-13.

    A GPU worker wedged in D state after a hardware fault. Cancelling it parked the one
    scheduler thread in ``process.join()`` forever, so no further task of any group was
    ever started -- while the API kept reporting the 8 idle CPU slots as free. Only
    restarting Apache helped.
    """

    def setUp(self):
        # keep the SIGTERM/SIGKILL escalation short enough for a test
        self._patches = [
            mock.patch.object(taskworkerthread, 'TASK_WORKER_TERMINATE_TIMEOUT', 0.1),
            mock.patch.object(taskworkerthread, '_REAP_POLL_S', 0.1),
            _only_gpu_workers_are_stuck(),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

        self.resources = Resources([
            TaskResource(TaskWorkerGroup.LONG_TASKS_GPU, 0),
            TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU),
            TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU),
        ])
        self.worker = OperationWorker(resources=self.resources, watcher_interval=-1)
        self.addCleanup(self.worker.shutdown)
        self.gpu_resource = self.resources.resources[0]

    def _start_stuck_gpu_task(self) -> str:
        task_id = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 600), None)
        self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.RUNNING))
        return task_id

    def test_cancelling_an_unkillable_worker_does_not_block_the_scheduler(self):
        gpu_task = self._start_stuck_gpu_task()

        self.worker.stop(gpu_task)

        # the whole point: the scheduler keeps ticking instead of parking in join()
        creator = self.worker.task_creator()
        iterations = creator._iterations
        self._wait_until(lambda: creator._iterations > iterations + 5, timeout=10)
        self.assertTrue(creator.is_alive())
        self.assertLess(creator.heartbeat_age(), 2.0)

        # ... and CPU tasks still start. Before the fix these stayed QUEUED forever.
        cpu_task = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 1), None)
        self._wait_until(lambda: self._status_is(self.worker, cpu_task, TaskStatusCodes.RUNNING),
                         timeout=60)

    def test_unkillable_worker_quarantines_only_its_own_slot(self):
        gpu_task = self._start_stuck_gpu_task()
        self.worker.stop(gpu_task)

        # the reaper gives up on the process and takes the slot out of service
        self._wait_until(lambda: self.gpu_resource.quarantined, timeout=10)
        self.assertIn('survived SIGKILL', self.gpu_resource.quarantine_reason)

        # the CPU slots are untouched -- one wedged GPU must not cost the whole server
        self.assertEqual(2, n_free([TaskWorkerGroup.NORMAL_TASKS_CPU], self.worker))
        self.assertEqual(0, n_free([TaskWorkerGroup.LONG_TASKS_GPU], self.worker))
        self.assertEqual(1, n_workers([TaskWorkerGroup.LONG_TASKS_GPU], self.worker))

        # and nothing is scheduled onto the dead card
        next_gpu = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.LONG_TASKS_GPU], 1), None)
        time.sleep(1.0)
        self.assertEqual(TaskStatusCodes.QUEUED, self.worker.status(next_gpu).code)

    def test_releasing_the_slot_makes_it_usable_again(self):
        gpu_task = self._start_stuck_gpu_task()
        self.worker.stop(gpu_task)
        self._wait_until(lambda: self.gpu_resource.quarantined, timeout=10)

        # the operator escape hatch that used to require restarting Apache
        self.assertTrue(self.worker.release_resource(0))
        self.assertFalse(self.gpu_resource.quarantined)
        self.assertFalse(self.gpu_resource.used)
        self.assertEqual(1, n_free([TaskWorkerGroup.LONG_TASKS_GPU], self.worker))


class TestSchedulerSurvivesFailures(SchedulerTestCase):
    def setUp(self):
        self.resources = Resources([TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU)])
        self.worker = OperationWorker(resources=self.resources, watcher_interval=-1)
        self.addCleanup(self.worker.shutdown)

    def test_an_exception_in_the_loop_does_not_kill_the_scheduler(self):
        creator = self.worker.task_creator()
        real_list_queued = TaskQueue.list_queued
        calls = {'n': 0}

        def flaky(queue_self):
            calls['n'] += 1
            if calls['n'] <= 3:
                raise RuntimeError('boom')
            return real_list_queued(queue_self)

        with mock.patch.object(TaskQueue, 'list_queued', new=flaky):
            self._wait_until(lambda: calls['n'] > 3, timeout=10)

        self.assertTrue(creator.is_alive())
        task_id = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 1), None)
        self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.RUNNING),
                         timeout=60)

    def test_a_task_that_cannot_be_started_fails_instead_of_stalling_everything(self):
        creator = self.worker.task_creator()
        with mock.patch.object(taskworkerthread, '_WorkerProcess',
                               side_effect=OSError('cannot spawn')):
            task_id = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 1), None)
            self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.ERROR),
                             timeout=20)

        self.assertTrue(creator.is_alive())
        # the slot was never marked used, so it is available for the next task
        self.assertEqual(0, self.resources.n_used())


class TestWatchdog(SchedulerTestCase):
    def setUp(self):
        self.resources = Resources([TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU)])
        self.worker = OperationWorker(resources=self.resources, watcher_interval=-1)
        self.addCleanup(self.worker.shutdown)
        # a watcher we drive by hand: interval large enough that its own thread never fires
        self.watcher = TaskWatcher(self.resources, self.worker.queue, 3600,
                                   worker=self.worker, max_heartbeat_age=0.5)
        self.addCleanup(self.watcher.stop)

    def test_it_restarts_a_dead_scheduler(self):
        creator = self.worker.task_creator()
        # retire the loop the same way a restart does, i.e. leave the thread gone
        creator._generation += 1
        self._wait_until(lambda: not creator.is_alive(), timeout=10)

        task_id = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 1), None)
        time.sleep(0.5)
        self.assertEqual(TaskStatusCodes.QUEUED, self.worker.status(task_id).code)

        self.assertTrue(self.watcher.check_once()['creator_restarted'])
        self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.RUNNING),
                         timeout=60)

    def test_it_restarts_a_dead_communicator(self):
        communicator = self.worker.task_communicator()
        communicator._generation += 1
        self._wait_until(lambda: not communicator.is_alive(), timeout=10)

        self.assertTrue(self.watcher.check_once()['communicator_restarted'])
        self.assertTrue(communicator.is_alive())

    def test_a_healthy_scheduler_is_left_alone(self):
        self.worker.task_creator()
        self.worker.task_communicator()
        time.sleep(0.3)
        self.assertEqual({'creator_restarted': False, 'communicator_restarted': False},
                         self.watcher.check_once())


class TestReconcile(SchedulerTestCase):
    def setUp(self):
        self.resources = Resources([TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU)])
        self.worker = OperationWorker(resources=self.resources, watcher_interval=-1)
        self.addCleanup(self.worker.shutdown)

    def test_it_frees_a_slot_that_no_task_occupies(self):
        creator = self.worker.task_creator()
        self.resources.resources[0].used = True     # the leak a crashed scheduler leaves behind

        self.assertEqual(1, creator.reconcile())
        self.assertFalse(self.resources.resources[0].used)

    def test_it_fails_a_task_left_running_without_a_worker(self):
        creator = self.worker.task_creator()
        task_id = self.worker.put(SleepyTaskRunner([TaskWorkerGroup.NORMAL_TASKS_CPU], 600), None)
        self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.RUNNING))

        # drop the bookkeeping, as a restarted scheduler used to
        running = creator.tasks.snapshot()[0]
        creator.tasks.remove(running)

        creator.reconcile()
        self._wait_until(lambda: self._status_is(self.worker, task_id, TaskStatusCodes.ERROR),
                         timeout=20)


class TestSchedulerHealthReporting(SchedulerTestCase):
    def setUp(self):
        self.resources = Resources([TaskResource(TaskWorkerGroup.NORMAL_TASKS_CPU)])
        self.worker = OperationWorker(resources=self.resources, watcher_interval=-1)
        self.addCleanup(self.worker.shutdown)

    def test_free_slots_are_zero_while_the_scheduler_is_unhealthy(self):
        creator = self.worker.task_creator()
        self.assertEqual(1, n_free([TaskWorkerGroup.NORMAL_TASKS_CPU], self.worker))

        # a slot nobody can hand out is not free -- this is what the UI got wrong
        creator._last_heartbeat = 0
        self.assertEqual(0, n_free([TaskWorkerGroup.NORMAL_TASKS_CPU], self.worker))
        self.assertEqual(1, n_workers([TaskWorkerGroup.NORMAL_TASKS_CPU], self.worker))
        self.assertFalse(self.worker.health()['healthy'])

    def test_health_of_a_worker_that_never_ran_anything(self):
        health = self.worker.health()
        self.assertFalse(health['started'])
        self.assertTrue(health['healthy'])


if __name__ == '__main__':
    unittest.main()
