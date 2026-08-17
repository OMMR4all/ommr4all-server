from rest_framework.views import APIView
from .auth import require_global_permissions, DatabasePermissionFlag, RestAPIUser
from restapi.views.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.models.error import APIError, ErrorCodes
from rest_framework.response import Response
from restapi.operationworker.operationworker import operation_worker
import logging
logger = logging.getLogger(__name__)


class TasksView(APIView):
    @require_global_permissions(DatabasePermissionFlag.TASKS_LIST)
    def get(self, request):
        return Response([{'id': t.task_id,
                          'status': t.public_status().to_dict(),
                          'creator': RestAPIUser.from_user(t.creator).to_dict(),
                          'algorithmType': t.task_runner.algorithm_type.value,
                          'book': t.task_runner.selection.book.get_meta().to_dict(),
                          } for t in operation_worker.queue.tasks])


class BookTasksView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        # Running/queued tasks for a single book. Book-scoped (only needs book
        # READ), so a regular editor — not just users with the global TASKS_LIST
        # permission — can recover a running workflow's progress after a reload.
        return Response([{'id': t.task_id,
                          'status': t.public_status().to_dict(),
                          'algorithmType': t.task_runner.algorithm_type.value,
                          } for t in operation_worker.queue.tasks
                         if t.task_runner.selection.book.book == book])


class TaskView(APIView):
    @require_global_permissions(DatabasePermissionFlag.TASKS_LIST)
    def get(self, request, task_id):
        return Response(operation_worker.queue.status_of_task(task_id).to_dict())

    @require_global_permissions(DatabasePermissionFlag.TASKS_CANCEL)
    def delete(self, request, task_id):
        operation_worker.stop(task_id)
        return Response()
