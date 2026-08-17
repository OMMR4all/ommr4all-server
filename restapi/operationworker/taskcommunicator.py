import queue
import time
from typing import NamedTuple, Optional, Union
from collections import OrderedDict

from ommr4all.settings import TASK_SCHEDULER_STALL_TIMEOUT
from .task import Task, TaskStatus, TaskStatusCodes, TaskNotFoundException
from .taskqueue import TaskQueue
import multiprocessing as mp
import threading
import logging
logger = logging.getLogger(__name__)


class TaskCommunicationData(NamedTuple):
    task: Task
    status: TaskStatus
    data: Union[dict, Exception] = None


class TaskCommunicator:
    def __init__(self, task_queue: TaskQueue):
        self.task_queue: TaskQueue = task_queue
        # spawn context: the queue is shared with the spawned task worker
        # processes (see taskworkerthread.py)
        self.queue = mp.get_context('spawn').Queue()
        # ids of tasks that reached a terminal state, so that
        # TaskWorkerThread.finished() can tell a completed-and-popped task
        # apart from a canceled one even after it left the task queue
        self._terminal_task_ids = OrderedDict()
        self._generation = 0
        self._last_heartbeat = time.time()
        self._restarts = 0
        self.thread: Optional[threading.Thread] = None
        # use thread to be in same memory pool as task queue
        self._start_thread()

    def _start_thread(self):
        self._generation += 1
        self._last_heartbeat = time.time()
        self.thread = threading.Thread(target=self.run, args=(self._generation,), name='task_communicator')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def is_terminated(self, task_id: str) -> bool:
        return task_id in self._terminal_task_ids

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def heartbeat_age(self) -> float:
        return time.time() - self._last_heartbeat

    def healthy(self, max_age: float = TASK_SCHEDULER_STALL_TIMEOUT) -> bool:
        return self.is_alive() and self.heartbeat_age() <= max_age

    def stop_loop(self) -> None:
        """Retire the loop without starting a new one (see TaskCreator.stop_loop)."""
        self._generation += 1

    def ensure_alive(self, max_age: float = TASK_SCHEDULER_STALL_TIMEOUT) -> bool:
        """Restart the status pump if it died or stalled. True if it had to act.

        Without it no task status is ever applied again: tasks stay RUNNING forever and
        no resource is released -- the same end state as a dead scheduler."""
        if self.healthy(max_age):
            return False
        self._restarts += 1
        logger.error('THREAD task_communicator: restarting (alive={}, heartbeat age={:.1f}s)'.format(
            self.is_alive(), self.heartbeat_age()))
        self._start_thread()
        return True

    def state(self) -> dict:
        return {
            'alive': self.is_alive(),
            'heartbeatAge': round(self.heartbeat_age(), 3),
            'restarts': self._restarts,
        }

    def run(self, generation: int):
        logger.info("THREAD task_communicator: Started (generation {})".format(generation))
        while generation == self._generation:
            # the timeout turns the heartbeat into a real liveness signal instead of one
            # that only advances when a worker happens to send something
            try:
                com: TaskCommunicationData = self.queue.get(timeout=1.0)
            except queue.Empty:
                self._last_heartbeat = time.time()
                continue
            except (EOFError, OSError):
                self._last_heartbeat = time.time()
                continue
            except Exception as e:
                logger.exception('THREAD task_communicator: failed to read a status update: {}'.format(e))
                self._last_heartbeat = time.time()
                continue

            try:
                if com.status.code == TaskStatusCodes.FINISHED or com.status.code == TaskStatusCodes.ERROR:
                    # record before update_status: the result may be popped
                    # (removing the task) as soon as the status is visible
                    self._terminal_task_ids[com.task.task_id] = True
                    while len(self._terminal_task_ids) > 1000:
                        self._terminal_task_ids.popitem(last=False)
                self.task_queue.update_status(com.task.task_id, com.status, com.data)
            except TaskNotFoundException:
                pass
            except Exception as e:
                # never die: this thread is the only writer of task status
                logger.exception('THREAD task_communicator: failed to apply a status update: {}'.format(e))
            finally:
                self._last_heartbeat = time.time()

        logger.info("THREAD task_communicator: generation {} retired".format(generation))

