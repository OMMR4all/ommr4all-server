from typing import Optional, NamedTuple, Set
from .taskqueue import TaskQueue, TaskStatus
from .taskcommunicator import TaskCommunicator
from uuid import uuid4
from .taskworkerthread import TaskWorkerThread, TaskWorkerGroup
from .taskrunners.taskrunner import TaskRunner
from django.conf import settings


class TaskIDGenerator:
    def gen(self):
        return str(uuid4())


class TaskSettings(NamedTuple):
    num_threads: int
    groups: Set[TaskWorkerGroup]


task_settings = [
    # GPU tasks, but add also threads if no GPU can be used
    TaskSettings(len(settings.GPU_SETTINGS.available_gpus), {TaskWorkerGroup.LONG_TASKS_GPU}),
    TaskSettings(1, {TaskWorkerGroup.LONG_TASKS_GPU}),

    # CPU only tasks
    TaskSettings(2, {TaskWorkerGroup.LONG_TASKS_CPU, TaskWorkerGroup.NORMAL_TASKS_CPU, TaskWorkerGroup.SHORT_TASKS_CPU}),
    TaskSettings(2, {TaskWorkerGroup.NORMAL_TASKS_CPU, TaskWorkerGroup.SHORT_TASKS_CPU}),
    TaskSettings(4, {TaskWorkerGroup.SHORT_TASKS_CPU}),
]


class OperationWorker:
    def __init__(self):
        self.queue = TaskQueue()
        self.task_communicator = TaskCommunicator(self.queue)
        self.threads = sum(
            [
                [
                    TaskWorkerThread("g{}-s{}".format(g, i),
                                     self.queue,
                                     self.task_communicator.queue,
                                     s.groups) for i in range(s.num_threads)
                ] for g, s in enumerate(task_settings)
            ],
            [])

        self.id_generator = TaskIDGenerator()

    def id_by_task_runner(self, task_runner: TaskRunner):
        return self.queue.id_by_runner(task_runner)

    def stop(self, task_id: str):
        task = self.queue.remove(task_id)
        if task is not None:
            for i, t in enumerate(self.threads):
                if t.cancel_if_current_task(task):
                    break

    def put(self, task_runner: TaskRunner) -> str:
        task_id = self.id_generator.gen()
        self.queue.put(task_id, task_runner)
        return task_id

    def pop_result(self, task_id: str) -> dict:
        return self.queue.pop_result(task_id)

    def status(self, task_id) -> Optional[TaskStatus]:
        return self.queue.status(task_id)


operation_worker = OperationWorker()
