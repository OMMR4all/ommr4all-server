import atexit
import logging
import multiprocessing as mp
# multiprocessing.util registers an atexit _exit_function that JOINS all
# non-daemon child processes. It is normally imported lazily on the first
# Process.start(), i.e. AFTER our atexit.register(_shutdown_workers) below —
# atexit runs LIFO, so the join would then run before our terminate and block
# the interpreter on the idle worker (the "hangs after tests finish" bug).
# Importing it here pins the order: _exit_function first registered, ours last
# registered, ours runs first.
import multiprocessing.util  # noqa: F401
import pickle
import time
from typing import Dict, Optional

from ommr4all.settings import TASK_WORKER_IDLE_TIMEOUT
from .task import Task, TaskStatus, TaskStatusCodes
from .taskcommunicator import TaskCommunicator, TaskCommunicationData
from .taskqueue import TaskNotFoundException
from .taskresources import TaskResource

logger = logging.getLogger(__name__)

# Workers are spawned, not forked: a fresh interpreter can always initialize
# CUDA, no matter what the Django parent process has imported or initialized.
# The context is local so that pools forked elsewhere (preprocessing, database
# loading) keep the default start method.
_spawn_ctx = mp.get_context('spawn')

# grace period between "worker process died" and failing its task, so that a
# final status still queued in the communicator wins over the crash detection
_WORKER_DEATH_GRACE_S = 5.0


class _WorkerProcess:
    """A long-lived spawned worker bound to one TaskResource.

    It stays alive between tasks (keeping loaded models warm, see
    omr/steps/predictorcache.py) and exits after TASK_WORKER_IDLE_TIMEOUT
    seconds without work.
    """

    def __init__(self, gpu_id: int, com_queue):
        from . import taskworkermain
        self.work_queue = _spawn_ctx.Queue()
        self.process = _spawn_ctx.Process(
            target=taskworkermain.main,
            args=(self.work_queue, com_queue, gpu_id,
                  TASK_WORKER_IDLE_TIMEOUT if TASK_WORKER_IDLE_TIMEOUT > 0 else None),
            name='task_worker_gpu{}'.format(gpu_id) if gpu_id >= 0 else 'task_worker_cpu',
        )
        self.process.daemon = False     # workers fork pools of their own, daemons may not
        self.process.start()

    def alive(self) -> bool:
        return self.process.is_alive()

    def submit(self, task: Task):
        # pre-pickled so that the worker controls when the payload (containing
        # Django model instances) is deserialized: after its django.setup()
        self.work_queue.put(pickle.dumps(task))

    def terminate(self):
        self.process.terminate()
        self.process.join()


# one persistent worker per TaskResource, keyed by object identity (the
# resource list lives as long as the server process)
_workers: Dict[int, _WorkerProcess] = {}


def _shutdown_workers():
    for worker in _workers.values():
        if worker.alive():
            worker.terminate()


atexit.register(_shutdown_workers)


class TaskWorkerThread:
    def __init__(self, resource: TaskResource, task: Task, communicator: TaskCommunicator):
        self.resource = resource
        self.task = task
        self.communicator = communicator
        self.com_queue = communicator.queue
        self.task_queue = communicator.task_queue
        self._worker_dead_since: Optional[float] = None

        worker = _workers.get(id(resource))
        if worker is None or not worker.alive():
            worker = _WorkerProcess(resource.gpu_id, self.com_queue)
            _workers[id(resource)] = worker
        self.worker = worker
        worker.submit(task)

    def finished(self) -> bool:
        try:
            status = self.task_queue.status_of_task(self.task.task_id)
        except TaskNotFoundException:
            if self.communicator.is_terminated(self.task.task_id):
                # completed normally and the result was already popped
                return True
            # the task vanished from the queue while still assigned to the
            # worker: it was removed by a stop request whose OP_STOP message
            # has not been processed yet, so stop the worker explicitly
            self.cancel()
            return True

        if status.code == TaskStatusCodes.FINISHED or status.code == TaskStatusCodes.ERROR:
            return True

        if not self.worker.alive():
            # the worker died mid-task (crash, OOM kill, ...); give the
            # communicator a grace period to deliver a final status that may
            # still be queued, then fail the task and discard the worker
            if self._worker_dead_since is None:
                self._worker_dead_since = time.time()
            elif time.time() - self._worker_dead_since > _WORKER_DEATH_GRACE_S:
                logger.error('THREAD {}: worker process died while running the task'.format(self.task.task_id))
                if _workers.get(id(self.resource)) is self.worker:
                    del _workers[id(self.resource)]
                self.com_queue.put(TaskCommunicationData(
                    self.task, TaskStatus(TaskStatusCodes.ERROR), Exception('Worker process died')))
                return True

        return False

    def cancel(self) -> bool:
        if self.task is None:
            return False

        if _workers.get(id(self.resource)) is self.worker:
            del _workers[id(self.resource)]

        if self.worker.alive():
            logger.info('THREAD {}: Attempting to terminate worker'.format(self.worker.process.name))
            self.worker.terminate()
            logger.info('THREAD {}: Worker terminated'.format(self.worker.process.name))
            return True

        return False
