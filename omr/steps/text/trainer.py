from database.file_formats.performance.pageprogress import Locks
from omr.dataset.datafiles import LockState
from abc import ABC
from typing import List

from omr.steps.algorithm import AlgorithmTrainer


class TextTrainerBase(AlgorithmTrainer, ABC):
    @staticmethod
    def required_locks() -> List[LockState]:
        return [LockState(Locks.TEXT, True)]
