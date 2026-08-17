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
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

from ommr4all.settings import TASK_MAX_RUNTIME, TASK_WORKER_IDLE_TIMEOUT, TASK_WORKER_TERMINATE_TIMEOUT
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

    def shutdown(self, timeout: float = None) -> bool:
        """Stop the worker, escalating SIGTERM -> SIGKILL. True if the process is gone.

        Waits at most 2*timeout. A process in uninterruptible sleep (D state, e.g. a
        wedged GPU driver or a hung mount) accepts *neither* signal, so the caller must
        be able to give up: returning False and abandoning the process is the whole
        point. Blocking, so only ever called from a reaper thread or from atexit --
        never from the scheduler loop.
        """
        if timeout is None:
            timeout = TASK_WORKER_TERMINATE_TIMEOUT

        # a feeder thread still holding queued payloads would block interpreter shutdown
        # on a child that will never read them (same class of hang as the module header)
        try:
            self.work_queue.close()
            self.work_queue.cancel_join_thread()
        except Exception:
            pass

        if not self.process.is_alive():
            self.process.join(0)        # reap the zombie
            return True

        self.process.terminate()
        self.process.join(timeout)
        if self.process.is_alive():
            logger.warning('WORKER %s (pid %s): did not stop on SIGTERM, sending SIGKILL',
                           self.process.name, self.process.pid)
            self.process.kill()
            self.process.join(timeout)

        return not self.process.is_alive()

    def poll_dead(self) -> bool:
        """Non-blocking check whether an abandoned process has finally exited."""
        if self.process.is_alive():
            return False
        self.process.join(0)
        return True

    def describe(self) -> dict:
        # 'worker', not 'name': the system resources view refuses to report anything that
        # describes the host, and its guard checks key names
        return {'worker': self.process.name, 'pid': self.process.pid, 'alive': self.process.is_alive()}


# one persistent worker per TaskResource, keyed by TaskResource.key
_workers: Dict[str, _WorkerProcess] = {}

# workers that were handed to a reaper and have not died yet; reported by the admin view
_retired: List[Tuple['_WorkerProcess', str, float]] = []
_retired_lock = threading.Lock()

# how often a reaper re-checks a process that survived SIGKILL
_REAP_POLL_S = 5.0


def retire(worker: _WorkerProcess, reason: str, resource: Optional[TaskResource] = None) -> None:
    """Stop a worker without blocking the caller.

    Termination runs on its own short-lived daemon thread, because it can take
    arbitrarily long -- an unkillable process is exactly the case this exists for.
    One thread per retirement (they are rare, only cancels and crashes) rather than a
    shared reaper, so a single pathological process cannot delay the next cleanup.

    If the process survives SIGKILL the resource is quarantined: its hardware is still
    held by the zombie, so nothing new may be scheduled onto it. The reaper keeps
    watching and lifts the quarantine by itself once the process finally does exit.
    """
    entry = (worker, reason, time.time())
    with _retired_lock:
        _retired.append(entry)

    def reap():
        try:
            if worker.shutdown():
                logger.info('WORKER %s: terminated (%s)', worker.process.name, reason)
            else:
                quarantine_reason = ('worker process {} (pid {}) survived SIGKILL and may still hold '
                                     'this slot'.format(worker.process.name, worker.process.pid))
                if resource is not None:
                    resource.quarantine(quarantine_reason)
                while not worker.poll_dead():
                    time.sleep(_REAP_POLL_S)
                logger.warning('WORKER %s: exited late, releasing the slot', worker.process.name)
                # only lift *our* quarantine: the slot may have been released by an admin
                # and wedged again since, and that newer quarantine must survive
                if resource is not None and resource.quarantine_reason == quarantine_reason:
                    resource.release_quarantine()
        except Exception as e:
            logger.exception('Failed to reap worker %s: %s', worker.process.name, e)
        finally:
            with _retired_lock:
                if entry in _retired:
                    _retired.remove(entry)

    threading.Thread(target=reap, name='worker_reaper', daemon=True).start()


def unreaped_workers() -> List[dict]:
    """Workers that were asked to stop and have not exited yet (for the admin view)."""
    with _retired_lock:
        entries = list(_retired)
    return [{**worker.describe(), 'reason': reason, 'retiredAt': at} for worker, reason, at in entries]


def _shutdown_workers():
    # bounded: an unkillable worker must not hang the interpreter (and with it an
    # Apache graceful restart, which used to be the only way out of a wedged scheduler)
    for worker in list(_workers.values()):
        if worker.alive() and not worker.shutdown(timeout=2.0):
            logger.warning('WORKER %s: abandoned at shutdown', worker.process.name)


atexit.register(_shutdown_workers)


class TaskWorkerThread:
    def __init__(self, resource: TaskResource, task: Task, communicator: TaskCommunicator):
        self.resource = resource
        self.task = task
        self.communicator = communicator
        self.com_queue = communicator.queue
        self.task_queue = communicator.task_queue
        self._worker_dead_since: Optional[float] = None

        worker = _workers.get(resource.key)
        if worker is None or not worker.alive():
            worker = _WorkerProcess(resource.gpu_id, self.com_queue)
            _workers[resource.key] = worker
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
            self.cancel('task removed from the queue')
            return True

        if status.code == TaskStatusCodes.FINISHED or status.code == TaskStatusCodes.ERROR:
            return True

        if TASK_MAX_RUNTIME > 0 and self.task.started_at is not None \
                and time.time() - self.task.started_at > TASK_MAX_RUNTIME:
            logger.error('THREAD {}: task exceeded the maximum runtime of {}s, stopping it'.format(
                self.task.task_id, TASK_MAX_RUNTIME))
            self.com_queue.put(TaskCommunicationData(
                self.task, TaskStatus(TaskStatusCodes.ERROR),
                Exception('Task exceeded the maximum runtime of {}s'.format(TASK_MAX_RUNTIME))))
            self.cancel('maximum runtime exceeded')
            return True

        if not self.worker.alive():
            # the worker died mid-task (crash, OOM kill, ...); give the
            # communicator a grace period to deliver a final status that may
            # still be queued, then fail the task and discard the worker
            if self._worker_dead_since is None:
                self._worker_dead_since = time.time()
            elif time.time() - self._worker_dead_since > _WORKER_DEATH_GRACE_S:
                logger.error('THREAD {}: worker process died while running the task'.format(self.task.task_id))
                if _workers.get(self.resource.key) is self.worker:
                    del _workers[self.resource.key]
                self.com_queue.put(TaskCommunicationData(
                    self.task, TaskStatus(TaskStatusCodes.ERROR), Exception('Worker process died')))
                return True

        return False

    def cancel(self, reason: str = 'canceled') -> bool:
        """Stop the worker. Returns immediately -- the actual termination is handed to a
        reaper thread, because it may never complete (see retire()). Blocking here is what
        let a single wedged GPU worker freeze the whole scheduler."""
        if self.task is None:
            return False

        if _workers.get(self.resource.key) is self.worker:
            del _workers[self.resource.key]

        if self.worker.alive():
            logger.info('THREAD {}: retiring worker ({})'.format(self.worker.process.name, reason))
            retire(self.worker, reason, self.resource)
            return True

        return False
