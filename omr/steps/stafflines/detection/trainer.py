from abc import ABC
from typing import List

from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmTrainer


class StaffLineDetectionTrainer(AlgorithmTrainer, ABC):
    @staticmethod
    def required_locks() -> List[LockState]:
        return [LockState(Locks.STAFF_LINES, True)]
