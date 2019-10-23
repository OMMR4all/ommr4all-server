from typing import List
from abc import ABC

from database.file_formats.performance import LockState
from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmTrainer


class SymbolDetectionTrainer(AlgorithmTrainer, ABC):
    @staticmethod
    def required_locks() -> List[LockState]:
        return [LockState(Locks.SYMBOLS, True)]
