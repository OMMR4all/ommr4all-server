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
from restapi.operationworker.taskrunners.taskrunner import TaskRunner
from restapi.operationworker.taskrunners.pageselection import PageSelection, PageSelectionParams
from omr.dataset.datafiles import EmptyDataSetException
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams
from omr.steps.step import Step, AlgorithmTypes
from dataclasses import field, dataclass

logger = logging.getLogger(__name__)


class BookPageSelectionView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, operation):
        body = json.loads(request.body, encoding='utf-8')
        book = DatabaseBook(book)
        algorithm = Step.predictor(AlgorithmTypes(operation))
        page_selection = PageSelection.from_params(PageSelectionParams.from_dict(body), book)
        pages = page_selection.get_pages(algorithm.unprocessed)
        return Response({
            'pages': [p.page for p in pages],
            'pageCount': page_selection.page_count.value,
            'singlePage': page_selection.single_page,
            'book': book.book,
            'totalPages': len(book.pages()),
        })


class BookOperationTaskView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, operation, task_id):
        op_status = operation_worker.status(task_id)
        if op_status:
            return Response({'status': op_status.to_dict()})
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
                result['status'] = op_status.to_dict()
                return Response(result)
            elif op_status.code == TaskStatusCodes.ERROR:
                error = operation_worker.pop_result(task_id)
                if isinstance(error, Exception):
                    raise error
                logger.error("Error in task: {} (status: )".format(error, op_status))
                return Response(error, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                return Response({'status': op_status.to_dict()})
        except TaskNotFoundException as e:
            return APIError(status.HTTP_404_NOT_FOUND,
                            "Task was not found.",
                            "Task not found.",
                            ErrorCodes.OPERATION_TASK_NOT_FOUND,
                            ).response()
        except KeyError as e:
            logger.exception(e)
            return APIError(status.HTTP_400_BAD_REQUEST,
                            "Invalid request. See server error logs for further information.",
                            "Invalid request",
                            ErrorCodes.OPERATION_INVALID_GET,
                            ).response()
        except (FileNotFoundError, OSError, IOError) as e:
            logger.exception(e)
            return APIError(status.HTTP_400_BAD_REQUEST,
                            "Model not found",
                            "No model was found",
                            ErrorCodes.OPERATION_TASK_NO_MODEL).response()
        except EmptyDataSetException as e:
            logger.exception(e)
            return APIError(status.HTTP_400_BAD_REQUEST,
                            "Dataset is empty",
                            "No ground truth files available",
                            ErrorCodes.OPERATION_TASK_TRAIN_EMPTY_DATASET,
                            ).response()
        except Exception as e:
            logging.exception(e)
            return APIError(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            str(e),
                            "Unknown server error",
                            ErrorCodes.OPERATION_UNKNOWN_SERVER_ERROR,
                            ).response()


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
                return Response({'status': op_status.to_dict()})
            else:
                return Response(status.HTTP_204_NO_CONTENT)

        return Response(status=status.HTTP_204_NO_CONTENT)


@dataclass()
class AlgorithmRequest(DataClassDictMixin):
    params: AlgorithmPredictorParams = field(default_factory=lambda: AlgorithmPredictorParams())
    selection: PageSelectionParams = field(default_factory=lambda: PageSelectionParams())


class BookOperationView(APIView):
    @staticmethod
    def op_to_task_runner(operation: str, book: DatabaseBook, body: dict) -> TaskRunner:
        from omr.steps.algorithmtypes import AlgorithmTypes
        for at in AlgorithmTypes:
            if at.value == operation:
                from restapi.operationworker.taskrunners.taskrunnerprediction import TaskRunnerPrediction, AlgorithmPredictorParams, Settings
                r = AlgorithmRequest.from_dict(body)
                return TaskRunnerPrediction(at,
                                            PageSelection.from_params(r.selection, book),
                                            Settings(r.params, store_to_pcgts=True)
                                            )
        # check if operation is linked to a task
        if operation == 'train_symbols':
            from restapi.operationworker.taskrunners.taskrunnersymboldetectiontrainer import TaskRunnerSymbolDetectionTrainer, TaskTrainerParams
            return TaskRunnerSymbolDetectionTrainer(book, TaskTrainerParams.from_dict(body.get('trainParams', {})))
        elif operation == 'train_staff_line_detector':
            from restapi.operationworker.taskrunners.taskrunnerstafflinedetectiontrainer import TaskRunnerStaffLineDetectionTrainer, TaskTrainerParams
            return TaskRunnerStaffLineDetectionTrainer(book, TaskTrainerParams.from_dict(body.get('trainParams', {})))
        else:
            raise NotImplementedError()

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def put(self, request, book, operation):
        body = json.loads(request.body, encoding='utf-8')
        book = DatabaseBook(book)
        task_runner = BookOperationView.op_to_task_runner(operation, book, body)
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


class BookOperationModelsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, operation):
        book = DatabaseBook(book)
        body = json.loads(request.body, encoding='utf-8') if request.body else {}
        task_runner = BookOperationView.op_to_task_runner(operation, book, body)
        models = task_runner.list_available_models_for_book(book)

        return Response(models.to_dict())


class BookOperationModelView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def delete(self, request, book, operation, model):
        book = DatabaseBook(book)
        task_runner = BookOperationView.op_to_task_runner(operation, book, {})
        # check that the model is really part of the model
        for m in task_runner.algorithm_meta().models_for_book(book).list_models():
            if m.id() == model:
                m.delete()
                return Response()

        return APIError(status.HTTP_404_NOT_FOUND,
                        "Model with id '{}' was not found in book '{}'. Available books {}.".format(
                            model, book.book,
                            [m.id() for m in task_runner.algorithm_meta().models_for_book(book).list_models()]),
                        "Model not found.",
                        ErrorCodes.MODEL_NOT_FOUND,
                        ).response()
