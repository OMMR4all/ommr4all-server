from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import authentication, permissions
from main.book import Page, Book, File, file_definitions, InvalidFileNameException
from omr.datatypes.performance.pageprogress import PageProgress
from omr.datatypes.performance.statistics import Statistics
from omr.datatypes.pcgts import PcGts
import json
from json import JSONDecodeError
import logging
import re
logger = logging.getLogger(__name__)


class PageProgressView(APIView):
    # authentication_classes = (authentication.TokenAuthentication,)
    # permission_classes = (permissions.IsAdminUser,)

    def get(self, request, book, page, format=None):
        page = Page(Book(book), page)
        file = File(page, 'page_progress')

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
        page = Page(Book(book), page)
        file = File(page, 'pcgts')

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
        page = Page(Book(book), page)
        file = File(page, 'statistics')

        if not file.exists():
            file.create()

        try:
            return Response(Statistics.from_json_file(file.local_path()).to_json())
        except JSONDecodeError as e:
            logging.error(e)
            file.delete()
            file.create()
            return Response(Statistics.from_json_file(file.local_path()).to_json())

