from typing import NamedTuple, Union
from collections import OrderedDict
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
        # use thread to be in same memory pool as task queue
        self.thread = threading.Thread(target=self.run, args=(), name='task_communicator')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def is_terminated(self, task_id: str) -> bool:
        return task_id in self._terminal_task_ids

    def run(self):
        logger.info("THREAD task_communicator: Started")
        while True:
            try:
                com: TaskCommunicationData = self.queue.get()
                if com.status.code == TaskStatusCodes.FINISHED or com.status.code == TaskStatusCodes.ERROR:
                    # record before update_status: the result may be popped
                    # (removing the task) as soon as the status is visible
                    self._terminal_task_ids[com.task.task_id] = True
                    while len(self._terminal_task_ids) > 1000:
                        self._terminal_task_ids.popitem(last=False)
                self.task_queue.update_status(com.task.task_id, com.status, com.data)
            except TaskNotFoundException:
                pass
            except EOFError:
                pass
            except Exception as e:
                raise e

