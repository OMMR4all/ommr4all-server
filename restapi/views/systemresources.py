from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from restapi.operationworker.operationworker import operation_worker
from restapi.operationworker.task import TaskStatusCodes
from restapi.operationworker.taskworkergroup import TaskWorkerGroup
from restapi.systeminfo import cpu_and_memory, cuda_status, disk_usage, gpu_stats, process_state
from .auth import DatabasePermissionFlag, RestAPIUser, require_admin


def _worker_pid(resource):
    """The pid of the process currently bound to a worker slot, or None."""
    from restapi.operationworker.taskworkerthread import _workers
    worker = _workers.get(resource.key)
    return worker.process.pid if worker is not None else None


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
    owns the task scheduler — the same caveat as /api/tasks.

    Administrators only, and even they get relative load figures only: see restapi.systeminfo."""

    @require_admin(DatabasePermissionFlag.VIEW_SYSTEM_RESOURCES)
    def get(self, request):
        gpus, gpu_available = gpu_stats()
        # ?refresh_cuda=1 re-runs the (slow) torch probe, e.g. after a driver update
        cuda = cuda_status(refresh=request.query_params.get('refresh_cuda') == '1')

        workers = [{
            'group': TaskWorkerGroup(resource.group).name.lower(),
            'gpu_id': resource.gpu_id,
            'used': bool(resource.used),
            'quarantined': bool(resource.quarantined),
            'quarantine_reason': resource.quarantine_reason,
            'quarantined_since': resource.quarantined_since,
            'process_state': process_state(_worker_pid(resource)),
            'task': _task_of_resource(resource) if resource.used else None,
        } for resource in operation_worker.resources.resources]

        queue_status = operation_worker.queue.status()

        return Response({
            **cpu_and_memory(),
            'disks': disk_usage(),
            'gpus': gpus,
            'gpu_available': gpu_available,
            'cuda': cuda,
            'workers': workers,
            # the slot flags alone cannot answer "can this server still start a task?"
            'scheduler': operation_worker.health(),
            'queue': {
                'n_total': queue_status.n_total,
                'n_running': queue_status.n_in_state.get(TaskStatusCodes.RUNNING, 0),
                'n_queued': queue_status.n_in_state.get(TaskStatusCodes.QUEUED, 0),
            },
        })


class SystemResourcesRepairView(APIView):
    """Recover the task scheduler in place.

    Exists because the only cure for a wedged scheduler used to be restarting the
    server: ``repair`` restarts dead or stalled scheduler threads and reclaims slots
    that were left marked busy, ``release`` force-frees a single slot whose worker
    process cannot be killed."""

    @require_admin(DatabasePermissionFlag.MANAGE_TASK_WORKERS)
    def post(self, request):
        action = request.data.get('action', 'repair')

        if action == 'repair':
            return Response(operation_worker.repair())

        if action == 'release':
            try:
                index = int(request.data.get('worker'))
            except (TypeError, ValueError):
                return Response({'error': "'worker' must be the index of a worker slot"},
                                status=status.HTTP_400_BAD_REQUEST)
            if not operation_worker.release_resource(index):
                return Response({'error': 'Unknown worker slot {}'.format(index)},
                                status=status.HTTP_400_BAD_REQUEST)
            return Response(operation_worker.health())

        return Response({'error': "Unknown action '{}' (expected 'repair' or 'release')".format(action)},
                        status=status.HTTP_400_BAD_REQUEST)
