from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from database import *
import logging
logger = logging.getLogger(__name__)


class BookCommentsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, book):
        from database.book_index import book_comments
        book = DatabaseBook(book)
        data = {
            'data': [{'comments': comments, 'page': page} for page, comments in book_comments(book)],
            'book': book.remote_path(),
        }

        return Response(data)


class BookCommentsCountView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, book):
        from database.book_index import book_comments_count
        return Response({'count': book_comments_count(DatabaseBook(book))})
