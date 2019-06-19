from rest_framework.views import APIView
from rest_framework import status
from database import DatabaseBook
from restapi.operationworker import operation_worker, TaskStatusCodes, \
    TaskNotFoundException, TaskAlreadyQueuedException
import logging
import json
from restapi.api.error import *
from restapi.api.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.api.pageaccess import require_lock
from restapi.operationworker.taskrunners.pageselection import PageSelection

logger = logging.getLogger(__name__)


class BookOperationTaskView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, operation, task_id):
        op_status = operation_worker.status(task_id)
        if op_status:
            return Response({'status': op_status.to_json()})
        else:
            return Response(status=status.HTTP_204_NO_CONTENT)

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def delete(self, request, book, operation, task_id):
        try:
            operation_worker.stop(task_id)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except TaskNotFoundException as e:
            logger.warning(e)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            logging.error(e)
            return Response({'error': 'unknown'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def post(self, request, book, operation, task_id):
        try:
            op_status = operation_worker.status(task_id)
            if op_status.code == TaskStatusCodes.FINISHED:
                result = operation_worker.pop_result(task_id)
                result['status'] = op_status.to_json()
                return Response(result)
            elif op_status.code == TaskStatusCodes.ERROR:
                error = operation_worker.pop_result(task_id)
                raise error
            else:
                return Response({'status': op_status.to_json()})
        except KeyError as e:
            logger.exception(e)
            return APIError(status.HTTP_400_BAD_REQUEST,
                            "Invalid request. See server error logs for further information.",
                            "Invalid request",
                            ErrorCodes.OPERATION_INVALID_GET,
                            ).response()
        except (FileNotFoundError, OSError) as e:
            logger.error(e)
            return Response({'error': 'no-model'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logging.error(e)
            return Response({'error': 'unknown'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BookOperationStatusView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, operation):
        book = DatabaseBook(book)
        body = json.loads(request.body, encoding='utf-8') if request.body else {}
        task_runner = BookOperationView.op_to_task_runner(operation, book, body)
        if task_runner is not None:
            task_id = operation_worker.id_by_task_runner(task_runner)
            op_status = operation_worker.status(task_id)
            if op_status:
                return Response({'status': op_status.to_json()})
            else:
                return Response(status.HTTP_204_NO_CONTENT)

        return Response(status=status.HTTP_204_NO_CONTENT)


class BookOperationView(APIView):
    @staticmethod
    def op_to_task_runner(operation, book: DatabaseBook, body: dict):
        # check if operation is linked to a task
        if operation == 'train_symbols':
            from restapi.operationworker.taskrunners.taskrunnersymboldetectiontrainer import TaskRunnerSymbolDetectionTrainer
            return TaskRunnerSymbolDetectionTrainer(book)
        elif operation == 'train_staff_line_detector':
            from restapi.operationworker.taskrunners.taskrunnerstafflinedetectiontrainer import TaskRunnerStaffLineDetectionTrainer
            return TaskRunnerStaffLineDetectionTrainer(book)
        elif operation == 'preprocessing':
            from restapi.operationworker.taskrunners.taskrunnerpreprocessing import TaskRunnerPreprocessing, Settings
            return TaskRunnerPreprocessing(
                PageSelection.from_json(body, book),
                Settings.from_json(body),
            )
        elif operation == 'stafflines':
            from restapi.operationworker.taskrunners.taskrunnerstafflinedetection import TaskRunnerStaffLineDetection, Settings
            return TaskRunnerStaffLineDetection(
                PageSelection.from_json(body, book),
                Settings(
                    store_to_pcgts=True,
                )
            )
        elif operation == 'layout':
            from restapi.operationworker.taskrunners.taskrunnerlayoutanalysis import TaskRunnerLayoutAnalysis, Settings
            return TaskRunnerLayoutAnalysis(
                PageSelection.from_json(body, book),
                Settings(
                    store_to_pcgts=True,
                )
            )
        elif operation == 'symbols':
            from restapi.operationworker.taskrunners.taskrunnersymboldetection import TaskRunnerSymbolDetection, Settings
            return TaskRunnerSymbolDetection(
                PageSelection.from_json(body, book),
                Settings(
                    store_to_pcgts=True,
                )
            )
        else:
            return None

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def put(self, request, book, operation):
        body = json.loads(request.body, encoding='utf-8')
        book = DatabaseBook(book)
        task_runner = BookOperationView.op_to_task_runner(operation, book, body)
        if task_runner:
            try:
                id = operation_worker.put(task_runner)
                return Response({'task_id': id}, status=status.HTTP_202_ACCEPTED)
            except TaskAlreadyQueuedException as e:
                return Response({'task_id': e.task_id}, status=status.HTTP_303_SEE_OTHER)
            except Exception as e:
                logger.error(e)
                return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, operation):
        book = DatabaseBook(book)
        body = json.loads(request.body, encoding='utf-8') if request.body else {}
        task_runner = BookOperationView.op_to_task_runner(operation, book, body)
        if task_runner is not None:
            task_id = operation_worker.id_by_task_runner(task_runner)
            if task_id:
                return Response({'task_id': task_id})

        return Response(status=status.HTTP_404_NOT_FOUND)
