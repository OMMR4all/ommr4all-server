import threading
from multiprocessing import Queue
import logging
from queue import Empty
import time

from ommr4all.settings import TASK_SCHEDULER_STALL_TIMEOUT
from .taskqueue import TaskQueue
from .taskcommunicator import TaskCommunicator, TaskCommunicationData
from typing import List, NamedTuple, Optional, Set
from .taskworkerthread import TaskWorkerThread
from .taskresources import Resources, TaskResource


logger = logging.getLogger(__name__)


class IntraComData(NamedTuple):
    op: int
    data: any


class _RunningTasks:
    """The tasks currently occupying a worker slot.

    Instance state rather than a local of the scheduler loop: the watchdog may restart
    that loop, and a restart that forgot which slots are busy would leak every one of
    them. The lock guards list mutation only -- never a process operation -- so it can
    never be held across a blocking call.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: List[TaskWorkerThread] = []

    def snapshot(self) -> List[TaskWorkerThread]:
        with self._lock:
            return list(self._tasks)

    def append(self, task: TaskWorkerThread):
        with self._lock:
            if task.resource.used:
                # not an assert: asserts vanish under -O, and this used to be able to
                # kill the scheduler thread outright
                logger.error('Resource of task {} is already in use, not scheduling it'.format(
                    task.task.task_id))
                return False
            task.resource.used = True
            # record the assignment on the queued task itself so that the
            # REST API can report which worker/GPU a running task occupies
            task.task.resource = task.resource
            self._tasks.append(task)
            logger.debug("Appended new task with id {} of type {}".format(
                task.task.task_id, type(task.task.task_runner)))
            return True

    def remove(self, task: TaskWorkerThread) -> bool:
        with self._lock:
            if task not in self._tasks:
                return False
            task.resource.used = False
            task.task.resource = None
            self._tasks.remove(task)
            logger.debug("Removed task with id {} of type {}".format(
                task.task.task_id, type(task.task.task_runner)))
            return True

    def cancel(self, task_id: str, reason: str = 'canceled') -> bool:
        for task in self.snapshot():
            if task.task.task_id == task_id:
                task.cancel(reason)
                logger.debug("Canceled task with id {} of type {}".format(
                    task.task.task_id, type(task.task.task_runner)))
                self.remove(task)
                return True

        return False

    def by_resource(self, resource: TaskResource) -> Optional[TaskWorkerThread]:
        for task in self.snapshot():
            if task.resource is resource:
                return task
        return None

    def resource_keys(self) -> Set[str]:
        return {task.resource.key for task in self.snapshot()}

    def task_ids(self) -> Set[str]:
        return {task.task.task_id for task in self.snapshot()}


class TaskCreator:
    OP_STOP = 0

    def __init__(self, task_queue: TaskQueue, task_communicator: TaskCommunicator, resources: Resources):
        self.task_queue: TaskQueue = task_queue
        self.task_communicator: TaskCommunicator = task_communicator
        self.resources: Resources = resources
        self.sleep = 0.1
        self.intra_com = Queue()
        self.tasks = _RunningTasks()
        self._generation = 0
        self._last_heartbeat = time.time()
        self._iterations = 0
        self._restarts = 0
        self.thread: Optional[threading.Thread] = None
        self._start_thread()

    def _start_thread(self):
        self._generation += 1
        self._last_heartbeat = time.time()
        # daemon thread to stop automatically on shutdown. Named task_creator, not
        # task_communicator -- both threads used to carry the same name, which made the
        # logs of a stalled scheduler ambiguous.
        self.thread = threading.Thread(target=self.run, args=(self._generation,), name='task_creator')
        self.thread.daemon = True
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def heartbeat_age(self) -> float:
        return time.time() - self._last_heartbeat

    def healthy(self, max_age: float = TASK_SCHEDULER_STALL_TIMEOUT) -> bool:
        return self.is_alive() and self.heartbeat_age() <= max_age

    def restart(self) -> None:
        """Start a fresh scheduler loop and retire the current one.

        A Python thread cannot be killed, so a stalled one is retired by generation: it
        exits at its next iteration boundary, whenever it becomes runnable again. Until
        then it is parked in a blocking call and cannot schedule anything, so the two
        loops never compete -- and the shared _RunningTasks is locked either way.
        """
        self._restarts += 1
        logger.error('THREAD task_creator: restarting the scheduler (alive={}, heartbeat age={:.1f}s)'.format(
            self.is_alive(), self.heartbeat_age()))
        self._start_thread()
        self.reconcile()

    def stop_loop(self) -> None:
        """Retire the loop without starting a new one.

        For shutdown: a daemon thread that is still logging while the interpreter
        finalizes crashes it ("could not acquire lock for <stdout>").
        """
        self._generation += 1

    def ensure_alive(self, max_age: float = TASK_SCHEDULER_STALL_TIMEOUT) -> bool:
        """Restart the scheduler if it died or stalled. True if it had to act."""
        if self.healthy(max_age):
            return False
        self.restart()
        return True

    def reconcile(self) -> int:
        """Bring the resource flags back in line with the tasks that actually run.

        Recovers the state a crashed or restarted scheduler leaves behind: slots marked
        used with nobody running on them, and tasks stuck at RUNNING whose worker thread
        is gone. Returns the number of repairs.
        """
        repaired = 0
        busy_keys = self.tasks.resource_keys()
        for resource in self.resources.resources:
            if resource.used and resource.key not in busy_keys:
                logger.error('Resource {} (gpu {}) was marked used without a running task, freeing it'.format(
                    resource.group, resource.gpu_id))
                resource.used = False
                repaired += 1

        from .task import TaskStatusCodes, TaskStatus
        running_ids = self.tasks.task_ids()
        for task in list(self.task_queue.tasks):
            if task.task_status.code == TaskStatusCodes.RUNNING and task.task_id not in running_ids:
                logger.error('Task {} was RUNNING without a worker, failing it'.format(task.task_id))
                self.task_communicator.queue.put(TaskCommunicationData(
                    task, TaskStatus(TaskStatusCodes.ERROR), Exception('The task was lost by the scheduler')))
                repaired += 1

        return repaired

    def release_resource(self, resource: TaskResource, fail_task: bool = True) -> bool:
        """Force a worker slot back into service without restarting the server.

        The operator escape hatch for a worker that cannot be killed: the process is
        abandoned, the quarantine lifted and the slot handed back to the scheduler.
        """
        from .task import TaskStatusCodes, TaskStatus
        running = self.tasks.by_resource(resource)
        if running is not None:
            if fail_task:
                self.task_communicator.queue.put(TaskCommunicationData(
                    running.task, TaskStatus(TaskStatusCodes.ERROR),
                    Exception('The worker slot was released by an administrator')))
            running.cancel('released by an administrator')
            self.tasks.remove(running)

        resource.used = False
        resource.release_quarantine()
        logger.warning('Resource {} (gpu {}) was released by an administrator'.format(
            resource.group, resource.gpu_id))
        return True

    def state(self) -> dict:
        return {
            'alive': self.is_alive(),
            'heartbeatAge': round(self.heartbeat_age(), 3),
            'iterations': self._iterations,
            'restarts': self._restarts,
            'nRunning': len(self.tasks.snapshot()),
        }

    def stop(self, task):
        self.intra_com.put(IntraComData(TaskCreator.OP_STOP, task.task_id))

    def _drain_intra_com(self):
        while True:
            try:
                data: IntraComData = self.intra_com.get_nowait()
            except Empty:
                return
            if data.op == TaskCreator.OP_STOP:
                self.tasks.cancel(data.data)

    def _cleanup(self):
        """Free the resources of tasks that are done.

        Each task is checked on its own: a task whose state cannot be determined is
        dropped rather than allowed to pin its slot (and, before, the whole scheduler).
        """
        for task in self.tasks.snapshot():
            try:
                done = task.finished()
            except Exception as e:
                logger.exception('THREAD task_creator: cleanup of task {} failed, dropping it: {}'.format(
                    task.task.task_id, e))
                done = True
            if done:
                self.tasks.remove(task)

    def _schedule(self):
        from .task import TaskStatusCodes, TaskStatus

        for task in self.task_queue.list_queued():
            for tg in task.task_runner.task_group:
                available = [r for r in self.resources.resources if r.group == tg and r.schedulable()]
                if not available:
                    continue
                r = available[0]
                try:
                    # construct first, mark afterwards: a failing spawn used to leave the
                    # task RUNNING forever *and* take the scheduler down with it
                    worker_thread = TaskWorkerThread(r, task, self.task_communicator)
                except Exception as e:
                    logger.exception('THREAD task_creator: could not start task {}: {}'.format(
                        task.task_id, e))
                    # fail it here and not only through the communicator: the status update
                    # is asynchronous, and until it lands the task stays queued and is
                    # retried on every 100ms tick
                    task.task_status.code = TaskStatusCodes.ERROR
                    self.task_communicator.queue.put(TaskCommunicationData(
                        task, TaskStatus(TaskStatusCodes.ERROR), Exception('The task could not be started')))
                    break
                if self.tasks.append(worker_thread):
                    task.started_at = time.time()
                    task.task_status.code = TaskStatusCodes.RUNNING
                break

    def run(self, generation: int):
        logger.info("THREAD task_creator: Started (generation {})".format(generation))

        while generation == self._generation:
            try:
                self._drain_intra_com()
                self._cleanup()
                self._schedule()
            except Exception as e:
                # the loop must survive anything: it is the only thread that starts tasks,
                # for every worker group, and it cannot be restarted from inside itself
                logger.exception('THREAD task_creator: unhandled error in the scheduler loop: {}'.format(e))
            finally:
                self._last_heartbeat = time.time()
                self._iterations += 1

            time.sleep(self.sleep)

        logger.info("THREAD task_creator: generation {} retired".format(generation))
