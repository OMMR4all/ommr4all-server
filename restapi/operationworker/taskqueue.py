from typing import List, Optional, NamedTuple, Dict
from django.contrib.auth.models import User
from .task import Task, \
    TaskAlreadyQueuedException, TaskNotFinishedException, TaskNotFoundException, \
    TaskStatusCodes, TaskStatus
from .taskrunners.taskrunner import TaskRunner
from multiprocessing import Lock


class TaskQueueStatus(NamedTuple):
    n_total: int
    n_in_state: Dict[TaskStatusCodes, int]


class TaskQueue:
    def __init__(self):
        self.tasks: List[Task] = []
        self.mutex = Lock()
        
    def status(self) -> TaskQueueStatus:
        with self.mutex:
            return TaskQueueStatus(
                len(self.tasks),
                {c: len([t for t in self.tasks if t.task_status.code == c]) for c in TaskStatusCodes}
            )

    def remove(self, task_id: str) -> Optional[Task]:
        with self.mutex:
            for i, t in enumerate(self.tasks):
                if t.task_id == task_id:
                    del self.tasks[i]
                    return t

            return None

    def has(self, task_id: str, task_runner: TaskRunner):
        with self.mutex:
            for task in self.tasks:
                if task.task_id == task_id or self._id_by_runner(task_runner) == task.task_id:
                    return True

        return False

    def put(self, task_id: str, task_runner: TaskRunner, creator: User):
        with self.mutex:
            for task in self.tasks:
                if task.task_id == task_id or self._id_by_runner(task_runner) == task.task_id:
                    raise TaskAlreadyQueuedException(task.task_id)

            self.tasks.append(Task(task_id, task_runner, TaskStatus(code=TaskStatusCodes.QUEUED),
                                   task_result={},
                                   creator=creator,
                                   ))

    def pop_result(self, task_id: str) -> dict:
        with self.mutex:
            for i, t in enumerate(self.tasks):
                if t.task_id == task_id:
                    if t.task_status.code == TaskStatusCodes.QUEUED or t.task_status.code == TaskStatusCodes.RUNNING:
                        raise TaskNotFinishedException()

                    del self.tasks[i]
                    return t.task_result

            raise TaskNotFoundException()

    def status_of_task(self, task_id: str) -> TaskStatus:
        with self.mutex:
            for task in self.tasks:
                if task.task_id == task_id:
                    return task.task_status

            raise TaskNotFoundException()

    def update_status(self, task_id: str, status: TaskStatus, result: dict = None):
        with self.mutex:
            for task in self.tasks:
                if task.task_id == task_id:
                    task.task_status = status
                    if result:
                        task.task_result = result
                    return

            raise TaskNotFoundException()

    def list_queued(self) -> List[Task]:
        with self.mutex:
            return [task for task in self.tasks if task.task_status.code == TaskStatusCodes.QUEUED]

    def _id_by_runner(self, task_runner: TaskRunner) -> Optional[str]:
        for task in self.tasks:
            if task.task_runner == task_runner or (type(task.task_runner) == type(task_runner) and task.task_runner.identifier() == task_runner.identifier()):
                return task.task_id
        return None

    def id_by_runner(self, task_runner: TaskRunner) -> Optional[str]:
        with self.mutex:
            return self._id_by_runner(task_runner)
