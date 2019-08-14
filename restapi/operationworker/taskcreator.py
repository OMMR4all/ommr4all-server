import threading
from multiprocessing import Queue
import logging
from queue import Empty
import time

from .taskqueue import TaskQueue
from .taskcommunicator import TaskCommunicator
from typing import NamedTuple, List
from .taskworkerthread import TaskWorkerThread
from .taskresources import Resources


logger = logging.getLogger(__name__)


class IntraComData(NamedTuple):
    op: int
    data: any


class TaskCreator:
    OP_STOP = 0

    def __init__(self, task_queue: TaskQueue, task_communicator: TaskCommunicator, resources: Resources):
        self.task_queue: TaskQueue = task_queue
        self.task_communicator: TaskCommunicator = task_communicator
        self.resources: Resources = resources
        self.sleep = 0.1
        self.intra_com = Queue()
        self.thread = threading.Thread(target=self.run, args=(), name='task_communicator')
        self.thread.daemon = True       # daemon thread to stop automatically on shutdown
        self.thread.start()

    def is_alive(self):
        return self.thread.is_alive()

    def stop(self, task):
        self.intra_com.put(IntraComData(TaskCreator.OP_STOP, task.task_id))

    def run(self):
        from .task import TaskStatusCodes

        class TaskList:
            def __init__(self):
                self.tasks: List[TaskWorkerThread] = []

            def cleanup(self):
                for task in self.tasks[:]:
                    if task.finished():
                        self.remove(task)

            def cancel(self, task_id: str):
                for task in self.tasks:
                    if task.task.task_id == task_id:
                        task.cancel()
                        logger.debug("Canceled task with id {} of type {}".format(task.task.task_id, type(task.task.task_runner)))
                        self.remove(task)
                        return True

                return False

            def remove(self, task: TaskWorkerThread):
                if task not in self.tasks:
                    raise ValueError()

                task.resource.used = False
                self.tasks.remove(task)
                logger.debug("Removed task with id {} of type {}".format(task.task.task_id, type(task.task.task_runner)))

            def append(self, task: TaskWorkerThread):
                assert(not task.resource.used)
                task.resource.used = True
                self.tasks.append(task)
                logger.debug("Appended new task with id {} of type {}".format(task.task.task_id, type(task.task.task_runner)))

        logger.info("THREAD TaskCreator: Started")
        resources = self.resources
        tasks = TaskList()

        while True:
            # check for tasks
            try:
                data: IntraComData = self.intra_com.get_nowait()
                if data.op == TaskCreator.OP_STOP:
                    tasks.cancel(data.data)
            except Empty:
                pass

            # cleanup threads that are stopped or do not exist anymore to free resources
            tasks.cleanup()

            # check if new tasks are available
            queued = self.task_queue.list_queued()
            for task in queued:
                for tg in task.task_runner.task_group:
                    available_resources_for_group = [r for r in resources.resources if r.group == tg and not r.used]
                    if len(available_resources_for_group) > 0:
                        task.task_status.code = TaskStatusCodes.RUNNING
                        r = available_resources_for_group[0]
                        tasks.append(TaskWorkerThread(r, task, self.task_communicator.queue))
                        break

            time.sleep(self.sleep)

