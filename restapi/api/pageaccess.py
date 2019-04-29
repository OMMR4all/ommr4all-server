from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from database import *
from database.file_formats.performance.pageprogress import PageProgress
from database.file_formats.performance.statistics import Statistics
from database.file_formats.pcgts import PcGts
from restapi.api.error import *
from json import JSONDecodeError
import logging
import json

logger = logging.getLogger(__name__)


class PageLockView(APIView):
    def get(self, request, book, page):
        logger.info(request)
        page = DatabasePage(DatabaseBook(book), page)
        return Response({'locked': page.is_locked_by_user(request.user.username)})

    def put(self, request, book, page):
        body: dict = json.loads(request.body)
        page = DatabasePage(DatabaseBook(book), page)
        if page.is_locked() and not body.get('force', False):
            if page.is_locked_by_user(request.user.username):
                return Response({'locked': True})
            else:
                # locked by another user
                user = page.lock_user()
                if not user:
                    # unknown user, we can force it
                    pass
                else:
                    return Response({'locked': False, 'first_name': user.first_name, 'last_name': user.last_name, 'email': user.email})

        page.lock(request.user.username)
        return Response({'locked': True})

    def delete(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        if page.is_locked_by_user(request.user.username):
            page.release_lock()
            return Response()

        return PageNotLockedAPIError(status=status.HTTP_423_LOCKED)


class PageProgressView(APIView):
    # authentication_classes = (authentication.TokenAuthentication,)
    # permission_classes = (permissions.IsAdminUser,)

    def get(self, request, book, page, format=None):
        page = DatabasePage(DatabaseBook(book), page)
        file = DatabaseFile(page, 'page_progress')

        if not file.exists():
            file.create()

        try:
            return Response(PageProgress.from_json_file(file.local_path()).to_json())
        except JSONDecodeError as e:
            logging.error(e)
            file.delete()
            file.create()
            return Response(PageProgress.from_json_file(file.local_path()).to_json())


class PagePcGtsView(APIView):
    def get(self, request, book, page, format=None):
        page = DatabasePage(DatabaseBook(book), page)
        file = DatabaseFile(page, 'pcgts')

        if not file.exists():
            file.create()

        try:
            return Response(PcGts.from_file(file).to_json())
        except JSONDecodeError as e:
            logging.error(e)
            file.delete()
            file.create()
            return Response(PcGts.from_file(file).to_json())


class PageStatisticsView(APIView):
    def get(self, request, book, page, format=None):
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

