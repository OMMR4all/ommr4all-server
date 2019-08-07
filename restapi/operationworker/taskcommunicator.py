from typing import NamedTuple
from .task import Task, TaskStatus, TaskNotFoundException
from .taskqueue import TaskQueue
from multiprocessing import Queue
import threading
import logging

logger = logging.getLogger(__name__)


class TaskCommunicationData(NamedTuple):
    task: Task
    status: TaskStatus
    data: dict = None


class TaskCommunicator:
    def __init__(self, task_queue: TaskQueue):
        self.task_queue: TaskQueue = task_queue
        self.queue = Queue()
        self.thread = threading.Thread(target=self.run, args=(), name='task_communicator')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def run(self):
        logger.info("THREAD task_communicator: Started")
        while True:
            try:
                com: TaskCommunicationData = self.queue.get()
                self.task_queue.update_status(com.task.task_id, com.status, com.data)
            except TaskNotFoundException:
                pass
            except EOFError:
                pass
            except Exception as e:
                raise e

