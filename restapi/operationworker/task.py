import time
from enum import IntEnum
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Union
from mashumaro import DataClassDictMixin

if TYPE_CHECKING:
    from .taskrunners.taskrunner import TaskRunner
    from .taskresources import TaskResource
    from django.contrib.auth.models import User


class TaskNotFoundException(Exception):
    pass


class TaskNotFinishedException(Exception):
    pass


class TaskAlreadyQueuedException(Exception):
    def __init__(self, task_id: str):
        self.task_id = task_id


class TaskStatusCodes(IntEnum):
    QUEUED = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3
    NOT_FOUND = 4


class TaskProgressCodes(IntEnum):
    INITIALIZING = 0
    WORKING = 1
    FINALIZING = 2
    RESOLVING_DATA = 3
    LOADING_DATA = 4
    PREPARING_TRAINING = 5


@dataclass
class TaskStatus(DataClassDictMixin):
    code: TaskStatusCodes = TaskStatusCodes.NOT_FOUND
    progress_code: TaskProgressCodes = TaskProgressCodes.INITIALIZING
    progress: float = -1
    accuracy: float = -1
    early_stopping_progress: float = -1
    loss: float = -1
    n_processed: int = 0
    n_total: int = 0
    # while QUEUED: number of queued tasks ahead that compete for the same
    # worker groups (0 = next in line); -1 = not queued/unknown
    queue_position: int = -1
    # seconds spent waiting in the queue; grows while QUEUED and freezes once the
    # task starts. -1 = unknown.
    queued_time: float = -1
    # seconds spent executing; grows while RUNNING and freezes at the total
    # duration once the task finished or failed. -1 = not started yet.
    run_time: float = -1


@dataclass
class Task:
    task_id: str
    task_runner: 'TaskRunner'
    task_status: TaskStatus
    task_result: Union[dict, Exception]
    creator: 'User'
    # the worker slot the task currently occupies; assigned by the TaskCreator
    # when the task starts running and cleared when it finishes. Only set for
    # RUNNING tasks, and only meaningful in the process that owns the scheduler.
    resource: Optional['TaskResource'] = None
    # Wall-clock timestamps of the task's lifecycle. They live on the Task and not
    # on the TaskStatus because every worker-side publish constructs a brand new
    # TaskStatus, which would wipe anything stored inside it on each update.
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # last time the worker published anything for this task. Distinguishes "legitimately
    # long running" from "the worker stopped talking to us", which a hung process looks like.
    updated_at: Optional[float] = None

    def public_status(self, queue_position: int = -1) -> TaskStatus:
        """The task status as reported by the REST API.

        Timings are projected at read time (like queue_position) so that a running
        task reports a live duration without the worker having to publish one."""
        now = time.time()
        if self.created_at is None:
            queued_time = -1.0
        else:
            queued_time = (self.started_at if self.started_at is not None else now) - self.created_at
        if self.started_at is None:
            run_time = -1.0
        else:
            run_time = (self.finished_at if self.finished_at is not None else now) - self.started_at

        return replace(self.task_status,
                       queue_position=queue_position,
                       queued_time=max(queued_time, -1.0),
                       run_time=max(run_time, -1.0),
                       )
