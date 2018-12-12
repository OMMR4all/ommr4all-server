from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import authentication, permissions
from django.http import FileResponse
from main.book import Page, Book, File, file_definitions, InvalidFileNameException
import json
import logging
import re
import os
logger = logging.getLogger(__name__)


class BookView(APIView):
    # authentication_classes = (authentication.TokenAuthentication,)
    # permission_classes = (permissions.IsAdminUser,)

    def get(self, request, book, format=None):
        book = Book(book)
        pages = book.pages()
        return Response({'pages': sorted([{'label': page.page} for page in pages if page.is_valid()], key=lambda v: v['label'])})

    def delete(self, request, book, format=None):
        book = Book(book)
        book.delete()
        return Response()


class BookUploadView(APIView):
    def post(self, request, book, format=None):
        from PIL import Image
        book = Book(book)
        if not os.path.exists(book.local_path()):
            os.mkdir(book.local_path())

        for type, file in request.FILES.items():
            name = os.path.splitext(os.path.basename(file.name))[0].replace(" ", "_").replace("-", "_").replace('.', '_')
            page = Page(book, name)
            if not os.path.exists(page.local_path()):
                os.mkdir(page.local_path())

            type = file.content_type
            if not type.startswith('image/'):
                return Response(status=status.HTTP_400_BAD_REQUEST)

            try:
                img = Image.open(file.file, 'r').convert('RGB')
                original = File(page, 'color_original')
                img.save(original.local_path())

            except Exception as e:
                logging.exception(e)
                return Response(status=status.HTTP_400_BAD_REQUEST)

        return Response()


class BooksView(APIView):
    def put(self, request, format=None):
        book = json.loads(request.body, encoding='utf-8')
        if 'name' not in book:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        book_id = re.sub('[^\w]', '_', book['name'])

        from main.book_meta import BookMeta
        try:
            b = Book(book_id)
            if b.exists():
                return Response(status=status.HTTP_304_NOT_MODIFIED)

            if b.create(BookMeta(id=b.book, name=book['name'])):
                return Response(b.get_meta().to_json())
        except InvalidFileNameException as e:
            logging.error(e)
            return Response(str(e), status=status.HTTP_406_NOT_ACCEPTABLE)

        return Response(status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, format=None):
        # TODO: sort by in request
        books = Book.list_available_book_metas()
        return Response({'books': sorted([book.to_json() for book in books], key=lambda b: b['name'])})


class BookDownloaderView(APIView):
    def get(self, request, book, type, format=None):
        book = Book(book)
        if type == 'annotations.zip':
            import zipfile, io, os
            s = io.BytesIO()
            zf = zipfile.ZipFile(s, 'w')
            pages = book.pages()
            for page in pages:
                file_names = ['color_deskewed', 'binary_deskewed', 'pcgts', ]
                files = [page.file(f) for f in file_names]

                if any([not f.exists() for f in files]):
                    continue

                for file, fn in zip(files, file_names):
                    zf.write(file.local_path(), os.path.join(fn, page.page + file.ext()))

            zf.close()
            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.zip')

        return Response(status=status.HTTP_400_BAD_REQUEST)

