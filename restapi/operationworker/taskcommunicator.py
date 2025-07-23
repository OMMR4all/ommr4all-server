from typing import NamedTuple, Union
from .task import Task, TaskStatus, TaskNotFoundException
from .taskqueue import TaskQueue
import multiprocessing as mp
import threading
import logging
#mp.set_start_method('spawn')
#import torch
logger = logging.getLogger(__name__)
#try:
#    # torch fork problem workaround
#    torch.set_num_threads(1)
#except RuntimeError:
#    pass
class TaskCommunicationData(NamedTuple):
    task: Task
    status: TaskStatus
    data: Union[dict, Exception] = None


class TaskCommunicator:
    def __init__(self, task_queue: TaskQueue):
        self.task_queue: TaskQueue = task_queue
        self.queue = mp.Queue()
        # use thread to be in same memory pool as task queue
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

