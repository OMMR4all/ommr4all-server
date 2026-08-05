from rest_framework.response import Response
from rest_framework.views import APIView

from restapi.operationworker.operationworker import operation_worker
from restapi.operationworker.task import TaskStatusCodes
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.systeminfo import cpu_and_memory, cuda_status, disk_usage, gpu_stats
from .auth import DatabasePermissionFlag, RestAPIUser, require_global_permissions


def _task_of_resource(resource) -> dict:
    """The running task occupying a worker slot, or None."""
    for task in operation_worker.queue.tasks:
        if task.resource is resource:
            try:
                book = task.task_runner.selection.book.get_meta().name
            except Exception:
                book = None
            return {
                'id': task.task_id,
                'algorithmType': task.task_runner.algorithm_type.value,
                'book': book,
                'creator': RestAPIUser.from_user(task.creator).to_dict() if task.creator else None,
            }
    return None


class SystemResourcesView(APIView):
    """Host resources (CPU/RAM/disk/GPU) plus the occupancy of the task worker
    slots. Note that the worker/queue part only covers the server process that
    owns the task scheduler — the same caveat as /api/tasks."""

    @require_global_permissions(DatabasePermissionFlag.TASKS_LIST)
    def get(self, request):
        gpus, gpu_error = gpu_stats()
        # ?refresh_cuda=1 re-runs the (slow) torch probe, e.g. after a driver update
        cuda = cuda_status(refresh=request.query_params.get('refresh_cuda') == '1')

        workers = [{
            'group': TaskWorkerGroup(resource.group).name.lower(),
            'gpu_id': resource.gpu_id,
            'used': bool(resource.used),
            'task': _task_of_resource(resource) if resource.used else None,
        } for resource in operation_worker.resources.resources]

        queue_status = operation_worker.queue.status()

        return Response({
            **cpu_and_memory(),
            'disks': disk_usage(),
            'gpus': gpus,
            'gpu_error': gpu_error,
            'cuda': cuda,
            'workers': workers,
            'queue': {
                'n_total': queue_status.n_total,
                'n_running': queue_status.n_in_state.get(TaskStatusCodes.RUNNING, 0),
                'n_queued': queue_status.n_in_state.get(TaskStatusCodes.QUEUED, 0),
            },
        })
