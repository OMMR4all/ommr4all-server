from rest_framework.views import APIView
from rest_framework import status, permissions
from database import DatabasePage, DatabaseBook, DatabaseFile
from restapi.operationworker import operation_worker, TaskStatusCodes, \
    TaskNotFoundException, TaskAlreadyQueuedException, TaskStatus
import logging
import json
from restapi.operationworker.taskrunners.pageselection import PageSelection
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from restapi.models.error import *
from restapi.views.bookaccess import require_permissions, DatabaseBookPermissionFlag
from dataclasses import field

logger = logging.getLogger(__name__)


class OperationTaskView(APIView):
    permission_classes = [permissions.AllowAny]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page, operation, task_id):
        op_status = operation_worker.status(task_id)
        if op_status:
            return Response({'status': op_status.to_dict()})
        else:
            return Response(status=status.HTTP_204_NO_CONTENT)

    @require_permissions([DatabaseBookPermissionFlag.READ_EDIT])
    def delete(self, request, book, page, operation, task_id):
        try:
            operation_worker.stop(task_id)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except TaskNotFoundException as e:
            logger.warning(e)
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            logging.error(e)
            return Response({'error': 'unknown'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @require_permissions([DatabaseBookPermissionFlag.READ_EDIT])
    def post(self, request, book, page, operation, task_id):
        try:
            op_status = operation_worker.status(task_id)
            if op_status.code == TaskStatusCodes.FINISHED:
                result = operation_worker.pop_result(task_id)
                result['status'] = op_status.to_dict()
                return Response(result)
            elif op_status.code == TaskStatusCodes.ERROR:
                error = operation_worker.pop_result(task_id)
                raise error
            else:
                return Response({'status': op_status.to_dict()})
        except TaskNotFoundException:
            return Response({'status': TaskStatus().to_dict()})
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


class OperationStatusView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page, operation):
        page = DatabasePage(DatabaseBook(book), page)
        body = json.loads(request.body, encoding='utf-8') if request.body else {}
        task_runner = OperationView.op_to_task_runner(operation, page, body)
        if task_runner is not None:
            task_id = operation_worker.id_by_task_runner(task_runner)
            op_status = operation_worker.status(task_id)
            if op_status:
                return Response({'status': op_status.to_dict()})
            else:
                return Response(status.HTTP_204_NO_CONTENT)

        return Response(status=status.HTTP_204_NO_CONTENT)


class OperationView(APIView):
    permission_classes = [permissions.AllowAny]

    @dataclass()
    class AlgorithmRequest(DataClassDictMixin):
        params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())

    @staticmethod
    def op_to_task_runner(operation, page: DatabasePage, body: dict):
        # check if operation is linked to a task
        from omr.steps.algorithmtypes import AlgorithmTypes
        for at in AlgorithmTypes:
            if at.value == operation:
                from restapi.operationworker.taskrunners.taskrunnerprediction import TaskRunnerPrediction, AlgorithmPredictorParams, Settings
                r = OperationView.AlgorithmRequest.from_dict(body)
                if 'pcgts' in body:
                    page.pcgts_from_dict(body['pcgts'])

                meta = page.book.get_meta()
                meta.algorithmPredictorParams[at] = r.params
                return TaskRunnerPrediction(at,
                                            PageSelection.from_page(page),
                                            Settings(meta.algorithm_predictor_params(at), store_to_pcgts=False)
                                            )

        return None

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, page, operation):
        body = json.loads(request.body, encoding='utf-8')
        page = DatabasePage(DatabaseBook(book), page)
        task_runner = OperationView.op_to_task_runner(operation, page, body)
        if task_runner:
            try:
                id = operation_worker.id_by_task_runner(task_runner)
                return Response({'task_id': id}, status=status.HTTP_202_ACCEPTED)
            except Exception as e:
                logger.error(e)
                return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

    @require_permissions([DatabaseBookPermissionFlag.EDIT])
    def put(self, request, book, page, operation):
        body = json.loads(request.body, encoding='utf-8')
        page = DatabasePage(DatabaseBook(book), page)
        task_runner = OperationView.op_to_task_runner(operation, page, body)
        if task_runner:
            try:
                id = operation_worker.put(task_runner, request.user)
                return Response({'task_id': id}, status=status.HTTP_202_ACCEPTED)
            except TaskAlreadyQueuedException as e:
                return Response({'task_id': e.task_id}, status=status.HTTP_303_SEE_OTHER)
            except Exception as e:
                logger.error(e)
                return Response(str(e), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

    @require_permissions([DatabaseBookPermissionFlag.DELETE_PAGES])
    def delete(self, request, book, page, operation):
        page = DatabasePage(DatabaseBook(book), page)

        if operation == 'clean':
            for key, _ in DatabaseFile.file_definitions().items():
                if key != 'color_original':
                    DatabaseFile(page, key).delete()

            return Response()
        elif operation == 'delete':
            page.delete()
            return Response()
