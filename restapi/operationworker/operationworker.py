from typing import Optional, TYPE_CHECKING
from .taskqueue import TaskQueue, TaskStatus
from .taskcommunicator import TaskCommunicator
from uuid import uuid4
from .taskresources import Resources, default_resources
from .taskrunners.taskrunner import TaskRunner
import logging
from .taskcreator import TaskCreator
from .taskwatcher import TaskWatcher
from ommr4all.settings import TASK_OPERATION_WATCHER_SETTINGS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from django.contrib.auth.models import User


class TaskIDGenerator:
    def gen(self):
        return str(uuid4())


class OperationWorker:
    def __init__(self, resources: Resources = None, watcher_interval=TASK_OPERATION_WATCHER_SETTINGS.interval):
        self.queue = TaskQueue()
        self.resources = resources if resources else default_resources()
        self._task_communicator: Optional[TaskCommunicator] = None
        self._task_creator: Optional[TaskCreator] = None
        self.id_generator = TaskIDGenerator()
        if watcher_interval > 0:
            self.task_watcher = TaskWatcher(self.resources, self.queue, watcher_interval)

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

    def put(self, task_runner: TaskRunner, creator: 'User') -> str:
        self.task_creator()  # require creation
        task_id = self.id_generator.gen()
        self.queue.put(task_id, task_runner, creator)
        return task_id

    def pop_result(self, task_id: str) -> dict:
        return self.queue.pop_result(task_id)

    def status(self, task_id) -> Optional[TaskStatus]:
        return self.queue.status_of_task(task_id)


operation_worker = OperationWorker()
