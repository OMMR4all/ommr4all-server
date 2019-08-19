from rest_framework.views import APIView
from .auth import require_global_permissions, DatabasePermissionFlag, RestAPIUser
from .error import APIError, ErrorCodes
from rest_framework import status
from rest_framework.response import Response
from database.models.bookstyles import BookStyle, BookStyleSerializer
from restapi.operationworker.operationworker import operation_worker
import logging
logger = logging.getLogger(__name__)


class TasksView(APIView):
    @require_global_permissions(DatabasePermissionFlag.TASKS_LIST)
    def get(self, request):
        return Response([{'id': t.task_id,
                          'status': t.task_status.to_dict(),
                          'creator': RestAPIUser.from_user(t.creator).to_dict(),
                          'algorithmType': t.task_runner.algorithm_type.value,
                          'book': t.task_runner.selection.book.get_meta().to_dict(),
                          } for t in operation_worker.queue.tasks])


class TaskView(APIView):
    @require_global_permissions(DatabasePermissionFlag.TASKS_LIST)
    def get(self, request, task_id):
        return Response(operation_worker.queue.status_of_task(task_id).to_dict())

    @require_global_permissions(DatabasePermissionFlag.TASKS_CANCEL)
    def delete(self, request, task_id):
        operation_worker.stop(task_id)
        return Response()
