from typing import Optional, NamedTuple, Set, Dict, List
from .taskqueue import TaskQueue, TaskStatus
from .taskcommunicator import TaskCommunicator
from uuid import uuid4
from .taskworkergroup import TaskWorkerGroup
from .taskworkerthread import TaskWorkerThread
from .taskrunners.taskrunner import TaskRunner
import threading
from multiprocessing import Queue
from queue import Empty
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


class TaskIDGenerator:
    def gen(self):
        return str(uuid4())


@dataclass
class TaskResource:
    group: TaskWorkerGroup
    gpu_id: int = -1
    used: bool = False


Resources = List[TaskResource]


class IntraComData(NamedTuple):
    op: int
    data: any


class TaskCreator:
    OP_STOP = 0

    def __init__(self, task_queue: TaskQueue, task_communicator: TaskCommunicator, resources: Resources):
        self.task_queue: TaskQueue = task_queue
        self.task_communicator: TaskCommunicator = task_communicator
        self.resources = resources
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

            # cleanup threads that are stopped
            tasks.cleanup()

            # check if new tasks are available
            queued = self.task_queue.list_queued()
            for task in queued:
                for tg in task.task_runner.task_group:
                    available_resources_for_group = [r for r in resources if r.group == tg and r.used == False]
                    if len(available_resources_for_group) > 0:
                        task.task_status.code = TaskStatusCodes.RUNNING
                        r = available_resources_for_group[0]
                        tasks.append(TaskWorkerThread(r, task, self.task_communicator.queue))
                        break

            time.sleep(self.sleep)


def default_resources() -> Resources:
    import ommr4all.settings as settings
    return [
               TaskResource(TaskWorkerGroup.LONG_TASKS_GPU, i) for i in settings.GPU_SETTINGS.available_gpus
           ] + [
               TaskResource(g) for g in ([TaskWorkerGroup.LONG_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.NORMAL_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.SHORT_TASKS_CPU] * 4)
           ]


class OperationWorker:
    def __init__(self, resources: Resources = None):
        self.queue = TaskQueue()
        self.resources = resources if resources else default_resources()
        self._task_communicator: Optional[TaskCommunicator] = None
        self._task_creator: Optional[TaskCreator] = None
        self.id_generator = TaskIDGenerator()

    def task_communicator(self) -> TaskCommunicator:
        if not self._task_communicator:
            self._task_communicator = TaskCommunicator(self.queue)
        return self._task_communicator

    def task_creator(self) -> TaskCreator:
        if not self._task_creator:
            self._task_creator = TaskCreator(self.queue, self.task_communicator(), self.resources)
        return self._task_creator

    def id_by_task_runner(self, task_runner: TaskRunner):
        return self.queue.id_by_runner(task_runner)

    def stop(self, task_id: str):
        task = self.queue.remove(task_id)
        if task is not None:
            self.task_creator().stop(task)

    def put(self, task_runner: TaskRunner) -> str:
        self.task_creator()  # require creation
        task_id = self.id_generator.gen()
        self.queue.put(task_id, task_runner)
        return task_id

    def pop_result(self, task_id: str) -> dict:
        return self.queue.pop_result(task_id)

    def status(self, task_id) -> Optional[TaskStatus]:
        return self.queue.status(task_id)


operation_worker = OperationWorker()
