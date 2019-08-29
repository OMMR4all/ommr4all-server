from enum import IntEnum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
from mashumaro import DataClassDictMixin

if TYPE_CHECKING:
    from .taskrunners.taskrunner import TaskRunner
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


@dataclass
class Task:
    task_id: str
    task_runner: 'TaskRunner'
    task_status: TaskStatus
    task_result: Union[dict, Exception]
    creator: 'User'
