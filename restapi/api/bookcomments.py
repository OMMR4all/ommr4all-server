from rest_framework.views import APIView
from rest_framework.response import Response
from database import *
from database.file_formats import PcGts
import logging
logger = logging.getLogger(__name__)


class BookCommentsView(APIView):
    def get(self, request, book):
        book = DatabaseBook(book)

        def comments_of_book(b):
            for page in b.pages():
                pcgts = PcGts.from_file(page.file('pcgts'))
                if len(pcgts.page.comments.comments) > 0:
                    yield {'comments': pcgts.page.comments.to_json(), 'page': page.page}

        data = {'data': list(comments_of_book(book)), 'book': book.remote_path()}

        return Response(data)


class BookCommentsCountView(APIView):
    def get(self, request, book):
        book = DatabaseBook(book)
        return Response({'count': sum([len(PcGts.from_file(page.file('pcgts', create_if_not_existing=True)).page.comments.comments) for page in book.pages()])})
