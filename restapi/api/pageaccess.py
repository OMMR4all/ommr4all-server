from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from database import *
from database.file_formats.performance.pageprogress import PageProgress, Locks
from database.file_formats.performance.statistics import Statistics
from database.file_formats.pcgts import PcGts
from restapi.api.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.api.error import *
from json import JSONDecodeError
import logging
import json
import re

logger = logging.getLogger(__name__)


def require_lock(func):
    def wrapper(view, request, book, page, *args, **kwargs):
        page = DatabaseBook(book).page(page)
        if not page.is_locked_by_user(request.user):
            return PageNotLockedAPIError(status.HTTP_423_LOCKED).response()
        else:
            return func(view, request, book, page.page, *args, **kwargs)

    return wrapper


class PageLockView(APIView):
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        return Response({'locked': page.is_locked_by_user(request.user)})

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def put(self, request, book, page):
        body: dict = json.loads(request.body)
        page = DatabasePage(DatabaseBook(book), page)
        if page.is_locked() and not body.get('force', False):
            if page.is_locked_by_user(request.user):
                return Response({'locked': True})
            else:
                # locked by another user
                user = page.lock_user()
                if not user:
                    # unknown user, we can force it
                    pass
                else:
                    return Response({'locked': False, 'first_name': user.first_name, 'last_name': user.last_name, 'email': user.email})

        page.lock(request.user)
        return Response({'locked': True})

    @require_lock
    def delete(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        page.release_lock()
        return Response()


class PageProgressVerifyView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        pp = DatabasePage(DatabaseBook(book), page).page_progress()
        return Response(pp.locked[Locks.VERIFIED])

    @require_permissions([DatabaseBookPermissionFlag.VERIFY_PAGE])
    def put(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        pp = page.page_progress()
        if pp.verified_allowed():
            pp.locked[Locks.VERIFIED] = True
        else:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "All user locks must be set in order to allow verification.",
                            "Verfication not allowed, not all locks are set.",
                            ErrorCodes.PAGE_PROGRESS_VERIFICATION_REQUIRES_ALL_PROGRESS_LOCKS,
                            ).response()
        page.save_page_progress()
        return Response()

    @require_permissions([DatabaseBookPermissionFlag.VERIFY_PAGE])
    def delete(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        pp = page.page_progress()
        pp.locked[Locks.VERIFIED] = False
        page.save_page_progress()
        return Response()


class PageProgressView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        return Response(page.page_progress().to_dict())


class PagePcGtsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        file = DatabaseFile(page, 'pcgts')

        if not file.exists():
            file.create()

        try:
            pcgts = PcGts.from_file(file)
            return Response(pcgts.to_json())
        except JSONDecodeError as e:
            logger.error(e)
            file.delete()
            file.create()
            return Response(PcGts.from_file(file).to_json())


class PageStatisticsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        file = DatabaseFile(page, 'statistics')

        if not file.exists():
            file.create()

        try:
            return Response(Statistics.from_json_file(file.local_path()).to_json())
        except JSONDecodeError as e:
            logging.error(e)
            file.delete()
            file.create()
            return Response(Statistics.from_json_file(file.local_path()).to_json())


class PageRenameView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.RENAME_PAGES])
    def post(self, request, book, page):
        page = DatabaseBook(book).page(page)
        obj = json.loads(request.body, encoding='utf-8')
        name = obj['name']
        name = re.sub(r'[^\w]', '_', name)

        if name != obj['name']:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Renaming page not possible, because the new name '{}' is invalid: '{}' != '{}'".format(obj['name'], obj['name'], name),
                            "Invalid page name '{}'".format(obj['name']),
                            ErrorCodes.PAGE_INVALID_NAME,
                            ).response()

        try:
            page.rename(name)
        except InvalidFileNameException:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Renaming page not possible, because the new name '{}' is invalid: '{}' != '{}'".format(obj['name'], obj['name'], name),
                            "Invalid page name '{}'".format(obj['name']),
                            ErrorCodes.PAGE_INVALID_NAME,
                            ).response()
        except FileExistsException as e:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Renaming page not possible, because a file at '{}' already exists".format(e.filename),
                            "A file at '{}' already exists".format(e.filename),
                            ErrorCodes.PAGE_EXISTS,
                            ).response()

        return Response()


