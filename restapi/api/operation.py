from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from database import DatabasePage, DatabaseBook, DatabaseFile, InvalidFileNameException, FileExistsException
from restapi.operationworker import operation_worker, TaskStatusCodes, TaskNotFoundException
import logging
import datetime
import json
import zipfile
import re
from database.file_formats.pcgts import PcGts, Coords
from database.file_formats.performance.pageprogress import PageProgress
from database.file_formats.performance.statistics import Statistics
from omr.stafflines.json_util import json_to_line
from restapi.operationworker import \
    TaskDataStaffLineDetection, TaskDataSymbolDetectionTrainer, TaskDataSymbolDetection, \
    TaskDataLayoutAnalysis
from restapi.operationworker.operation_worker import TaskAlreadyQueuedException
from restapi.operationworker.operationlayoutextractconnectedcomponentsbyline import \
    TaskDataExtractLayoutConnectedComponentByLine
from restapi.api.error import *
from restapi.api.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.api.pageaccess import require_lock

logger = logging.getLogger(__name__)


class OperationTaskView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page, operation, task_id):
        op_status = operation_worker.status(task_id)
        if op_status:
            return Response({'status': op_status.to_json()})
        else:
            return Response(status=status.HTTP_204_NO_CONTENT)

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
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

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    @require_lock
    def post(self, request, book, page, operation, task_id):
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


class OperationStatusView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page, operation):
        page = DatabasePage(DatabaseBook(book), page)
        body = json.loads(request.body, encoding='utf-8') if request.body else {}
        task_data = OperationView.op_to_task_data(operation, page, body)
        if task_data is not None:
            task_id = operation_worker.id_by_task_data(task_data)
            op_status = operation_worker.status(task_id)
            if op_status:
                return Response({'status': op_status.to_json()})
            else:
                return Response(status.HTTP_204_NO_CONTENT)

        return Response(status=status.HTTP_204_NO_CONTENT)


class OperationView(APIView):
    @staticmethod
    def op_to_task_data(operation, page: DatabasePage, body: dict):
        # check if operation is linked to a task
        if operation == 'staffs':
            return TaskDataStaffLineDetection(page)
        elif operation == 'symbols':
            return TaskDataSymbolDetection(page)
        elif operation == 'train_symbols':
            return TaskDataSymbolDetectionTrainer(page.book)
        elif operation == 'layout':
            return TaskDataLayoutAnalysis(page)
        elif operation == 'layout_extract_cc_by_line':
            return TaskDataExtractLayoutConnectedComponentByLine(page, Coords.from_json(body['points']))
        else:
            return None

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    @require_lock
    def post(self, request, book, page, operation, format=None):
        page = DatabasePage(DatabaseBook(book), page)

        if operation == 'save_page_progress':
            obj = json.loads(request.body, encoding='utf-8')
            pp = PageProgress.from_json(obj)
            pp.to_json_file(page.file('page_progress').local_path())

            # add to backup archive
            with zipfile.ZipFile(page.file('page_progress_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('page_progress_{}.json'.format(datetime.datetime.now()), json.dumps(pp.to_json(), indent=2))

            logger.info('Successfully saved page progress file to {}'.format(page.file('page_progress').local_path()))

            return Response()
        elif operation == 'save_statistics':
            obj = json.loads(request.body, encoding='utf-8')
            total_stats = Statistics.from_json(obj)
            total_stats.to_json_file(page.file('statistics').local_path())

            # add to backup archive
            with zipfile.ZipFile(page.file('statistics_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('statistics_{}.json'.format(datetime.datetime.now()), json.dumps(total_stats.to_json(), indent=2))

            logger.info('Successfully saved statistics file to {}'.format(page.file('statistics').local_path()))

            return Response()
        elif operation == 'save':
            obj = json.loads(request.body, encoding='utf-8')
            pcgts = PcGts.from_json(obj, page)
            pcgts.to_file(page.file('pcgts').local_path())

            # add to backup archive
            with zipfile.ZipFile(page.file('pcgts_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('pcgts_{}.json'.format(datetime.datetime.now()), json.dumps(pcgts.to_json(), indent=2))

            logger.info('Successfully saved pcgts file to {}'.format(page.file('pcgts').local_path()))

            return Response()

        return Response(status=status.HTTP_400_BAD_REQUEST)

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def put(self, request, book, page, operation):
        body = json.loads(request.body, encoding='utf-8')
        page = DatabasePage(DatabaseBook(book), page)
        task_data = OperationView.op_to_task_data(operation, page, body)
        if task_data:
            try:
                id = operation_worker.put(task_data)
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
