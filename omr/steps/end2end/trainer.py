from abc import ABC
from typing import List

from database.file_formats.performance.pageprogress import Locks
from omr.dataset.datafiles import LockState
from omr.steps.algorithm import AlgorithmTrainer


class End2EndTrainerBase(AlgorithmTrainer, ABC):
    @staticmethod
    def required_locks() -> List[LockState]:
        # GT strings need both symbols and syllable connections
        return [LockState(Locks.SYMBOLS, True), LockState(Locks.TEXT, True)]
