import logging
import time
from dataclasses import dataclass
from .taskworkergroup import TaskWorkerGroup
from typing import List, Optional
from multiprocessing import Value
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskResource:
    """One worker slot.

    ``used`` is the scheduler's own bookkeeping: it is set when a task is assigned and
    cleared when the task leaves. ``quarantined`` is the opposite kind of statement --
    it means the slot cannot be trusted right now, because the worker process that last
    occupied it could not be killed and is presumably still holding the slot's hardware
    (a wedged GPU, an unreadable disk). A quarantined slot is skipped by the scheduler
    but leaves every other slot usable, which is the difference between losing one GPU
    and losing the whole server.
    """

    def __init__(self, group: TaskWorkerGroup, gpu_id: int = -1):
        # identity for the worker registry in taskworkerthread.py. Not id(): CPython
        # recycles it, so a collected Resources object could hand a stale worker to an
        # unrelated slot.
        self.key = uuid4().hex
        self.group = group
        self.gpu_id = gpu_id
        self._used = Value('b', False)
        self._quarantined = Value('b', False)
        # plain attributes: only ever read/written by the parent process (reaper thread,
        # watcher, REST views), unlike the flags above which are also touched from workers
        self.quarantined_since: Optional[float] = None
        self.quarantine_reason: Optional[str] = None

    @property
    def used(self):
        return self._used.value

    @used.setter
    def used(self, v: bool):
        self._used.value = v

    @property
    def quarantined(self):
        return self._quarantined.value

    def quarantine(self, reason: str):
        if not self._quarantined.value:
            self.quarantined_since = time.time()
            logger.error('Resource %s (gpu %d) quarantined: %s', self.group, self.gpu_id, reason)
        self.quarantine_reason = reason
        self._quarantined.value = True

    def release_quarantine(self):
        if self._quarantined.value:
            logger.info('Resource %s (gpu %d) released from quarantine', self.group, self.gpu_id)
        self._quarantined.value = False
        self.quarantined_since = None
        self.quarantine_reason = None

    def schedulable(self) -> bool:
        return not self.used and not self.quarantined


ResourcesList = List[TaskResource]


class Resources:
    def __init__(self, resources: ResourcesList = None):
        self.resources = resources if resources else []

    def free(self) -> ResourcesList:
        """The slots a task may be scheduled on -- quarantined slots are not free."""
        return [r for r in self.resources if r.schedulable()]

    def used(self) -> ResourcesList:
        return [r for r in self.resources if r.used]

    def quarantined(self) -> ResourcesList:
        return [r for r in self.resources if r.quarantined]

    def n_free(self) -> int:
        return len(self.free())

    def n_used(self) -> int:
        return len(self.used())

    def n_quarantined(self) -> int:
        return len(self.quarantined())

    def n_total(self) -> int:
        return len(self.resources)


def resolve_available_gpus(configured: Optional[List[int]]) -> List[int]:
    """The GPU indices to register a worker slot for.

    An explicit configuration (OMMR4ALL_GPUS) always wins, including the empty list
    meaning "no GPU workers". Without one the cards are detected, so that a host with
    more than one GPU does not silently run everything on the first one.
    """
    if configured is not None:
        return list(configured)

    # nvidia-smi rather than torch: the Django parent that owns the scheduler pins
    # CUDA_VISIBLE_DEVICES='' and must never import torch. gpu_stats() already handles
    # the timeout, the env fixup and the caching.
    from restapi.systeminfo import gpu_stats
    gpus, available = gpu_stats()
    if not available:
        # no nvidia-smi at all: keep the historical single slot rather than silently
        # dropping GPU support on a host where the driver just is not queryable
        logger.info('GPUs could not be detected; registering a single GPU worker for device 0. '
                    'Set OMMR4ALL_GPUS to configure this explicitly.')
        return [0]

    detected = [g['index'] for g in gpus if g.get('index') is not None]
    logger.info('Detected %d GPU(s): %s', len(detected), detected)
    return detected


def default_resources() -> Resources:
    import ommr4all.settings as settings
    return Resources([
               TaskResource(TaskWorkerGroup.LONG_TASKS_GPU, i)
               for i in resolve_available_gpus(settings.GPU_SETTINGS.available_gpus)
           ] + [
               TaskResource(g) for g in ([TaskWorkerGroup.LONG_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.NORMAL_TASKS_CPU] * 2 +
                                         [TaskWorkerGroup.SHORT_TASKS_CPU] * 4)
           ])

