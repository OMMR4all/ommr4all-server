"""Policy glue between the per-algorithm worker resource policy (AlgorithmMeta)
and the task scheduler's TaskWorkerGroup resources.

The user may request a WorkerResource ('cpu'/'gpu') when starting a task. An
explicit request is strict: it is validated against the algorithm's allowed
set and never degraded — the task waits for a matching resource. An absent
request resolves to the algorithm's default, degraded to another allowed
resource only if no worker exists for the default at all (e.g. training on a
GPU-less server).
"""
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from omr.steps.algorithmtypes import AlgorithmTypes, WorkerResource
from .taskworkergroup import TaskWorkerGroup

if TYPE_CHECKING:
    from omr.steps.algorithm import AlgorithmMeta


class InvalidWorkerResourceException(Exception):
    pass


# training operation name (REST path segment) -> the algorithm that is trained
TRAIN_OPERATIONS: Dict[str, AlgorithmTypes] = {
    'train_symbols': AlgorithmTypes.SYMBOLS_PC_TORCH,
    'train_staff_line_detector': AlgorithmTypes.STAFF_LINES_PC_Torch,
    'train_layout_detector': AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL_YOLO,
    'train_character_recognition': AlgorithmTypes.OCR_GUPPY,
    'train_end2end': AlgorithmTypes.END2END_SWIN,
}


def groups_for(resource: WorkerResource, training: bool) -> List[TaskWorkerGroup]:
    # only LONG_TASKS_GPU resources carry a gpu_id >= 0 (see taskresources.py),
    # so any GPU task must target that group
    if resource == WorkerResource.GPU:
        return [TaskWorkerGroup.LONG_TASKS_GPU]
    return [TaskWorkerGroup.LONG_TASKS_CPU] if training else [TaskWorkerGroup.NORMAL_TASKS_CPU]


def _worker(worker=None):
    if worker is not None:
        return worker
    from . import operation_worker
    return operation_worker


def n_workers(groups: List[TaskWorkerGroup], worker=None) -> int:
    return len([r for r in _worker(worker).resources.resources if r.group in groups])


def n_free(groups: List[TaskWorkerGroup], worker=None) -> int:
    return len([r for r in _worker(worker).resources.resources if r.group in groups and not r.used])


def n_queued(groups: List[TaskWorkerGroup], worker=None) -> int:
    return len([t for t in _worker(worker).queue.list_queued()
                if set(t.task_runner.task_group) & set(groups)])


def resolve_worker_resource(meta: Type['AlgorithmMeta'],
                            requested: Optional[str],
                            training: bool,
                            worker=None) -> WorkerResource:
    allowed = meta.allowed_trainer_resources() if training else meta.allowed_predictor_resources()
    default = meta.default_trainer_resource() if training else meta.default_predictor_resource()

    if requested:
        try:
            resource = WorkerResource(requested)
        except ValueError:
            raise InvalidWorkerResourceException(
                "Unknown worker resource '{}' (expected one of {})".format(
                    requested, [r.value for r in WorkerResource]))
        if resource not in allowed:
            raise InvalidWorkerResourceException(
                "The algorithm '{}' only supports the worker resources {}".format(
                    meta.type().value, [r.value for r in allowed]))
        return resource

    if n_workers(groups_for(default, training), worker) == 0:
        for alternative in allowed:
            if n_workers(groups_for(alternative, training), worker) > 0:
                return alternative
    return default
