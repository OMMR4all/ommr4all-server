from enum import IntEnum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .taskrunners.taskrunner import TaskRunner


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


@dataclass
class TaskStatus:
    code: TaskStatusCodes = TaskStatusCodes.NOT_FOUND
    progress_code: TaskProgressCodes = TaskProgressCodes.INITIALIZING
    progress: float = -1
    accuracy: float = -1
    early_stopping_progress: float = -1
    loss: float = -1
    n_processed: int = 0
    n_total: int = 0

    def to_json(self):
        return {
            'code': self.code.value,
            'progress_code': self.progress_code.value,
            'progress': self.progress,
            'accuracy': self.accuracy,
            'early_stopping_progress': self.early_stopping_progress,
            'loss': self.loss,
            'n_processed': self.n_processed,
            'n_total': self.n_total,
        }


@dataclass
class Task:
    task_id: str
    task_runner: 'TaskRunner'
    task_status: TaskStatus
    task_result: dict

