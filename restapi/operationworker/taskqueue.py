import time
from dataclasses import replace
from typing import List, Optional, NamedTuple, Dict, TYPE_CHECKING
from .task import Task, \
    TaskAlreadyQueuedException, TaskNotFinishedException, TaskNotFoundException, \
    TaskStatusCodes, TaskStatus
from .taskrunners.taskrunner import TaskRunner
from multiprocessing import Lock

if TYPE_CHECKING:
    from django.contrib.auth.models import User


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
                if task.task_id == task_id:# Todo refactor id by runner to only use the id of the task
                    return True

        return False

    def put(self, task_id: str, task_runner: TaskRunner, creator: 'User'):
        with self.mutex:
            for task in self.tasks:
                # Todo refactor id by runner to only use the id of the task
                if task.task_id == task_id:# or self._id_by_runner(task_runner) == task.task_id:
                    raise TaskAlreadyQueuedException(task.task_id)

            self.tasks.append(Task(task_id, task_runner, TaskStatus(code=TaskStatusCodes.QUEUED),
                                   task_result={},
                                   creator=creator,
                                   created_at=time.time(),
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
                    if task.task_status.code == TaskStatusCodes.QUEUED:
                        groups = set(task.task_runner.task_group)
                        n_ahead = 0
                        for other in self.tasks:
                            if other.task_id == task_id:
                                break
                            if other.task_status.code == TaskStatusCodes.QUEUED \
                                    and groups & set(other.task_runner.task_group):
                                n_ahead += 1
                        return task.public_status(queue_position=n_ahead)
                    return task.public_status()

            raise TaskNotFoundException()

    @staticmethod
    def _monotonic(previous: TaskStatus, status: TaskStatus) -> TaskStatus:
        """Stop a running task's progress from moving backwards.

        Predictors report from more than one place (a per-page loop plus an ML
        library's own step callback, say). If those disagree on scale the bar
        visibly bounces, so hold the high-water mark instead of trusting the last
        writer. Only applies while the task keeps running in the same phase: a
        progress_code change (LOADING_DATA -> WORKING) legitimately restarts the
        scale, and a terminal status must always be published verbatim.
        """
        if status.code != TaskStatusCodes.RUNNING or previous.code != TaskStatusCodes.RUNNING:
            return status
        if status.progress_code != previous.progress_code:
            return status
        if status.progress >= previous.progress and status.n_processed >= previous.n_processed:
            return status
        return replace(status,
                       progress=max(status.progress, previous.progress),
                       n_processed=max(status.n_processed, previous.n_processed))

    def update_status(self, task_id: str, status: TaskStatus, result: dict = None):
        with self.mutex:
            for task in self.tasks:
                if task.task_id == task_id:
                    task.task_status = self._monotonic(task.task_status, status)
                    task.updated_at = time.time()
                    if task.finished_at is None and task.task_status.code in \
                            (TaskStatusCodes.FINISHED, TaskStatusCodes.ERROR):
                        task.finished_at = time.time()
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
