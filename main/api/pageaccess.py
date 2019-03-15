from rest_framework.views import APIView
from rest_framework.response import Response
from database import *
from database.file_formats.performance.pageprogress import PageProgress
from database.file_formats.performance.statistics import Statistics
from database.file_formats.pcgts import PcGts
from json import JSONDecodeError
import logging

logger = logging.getLogger(__name__)


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

