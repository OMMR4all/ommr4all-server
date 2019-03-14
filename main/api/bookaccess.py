from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
from main.book import Page, Book, File, InvalidFileNameException
from main.book_meta import BookMeta
import json
import logging
import re
import os
logger = logging.getLogger(__name__)


class BookMetaView(APIView):
    def get(self, request, book, format=None):
        book = Book(book)
        return Response(book.get_meta().to_json())

    def put(self, request, book, format=None):
        book = Book(book)
        meta = BookMeta.from_json(book, json.loads(request.body, encoding='utf-8'))
        book.save_json_to_meta(meta.to_json())
        return Response()


class BookView(APIView):
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
            logger.debug('Received new image of content type {}'.format(file.content_type))
            name = os.path.splitext(os.path.basename(file.name))[0]
            name = re.sub('[^\w]', '_', name)
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
                logger.debug('Created page at {}'.format(page.local_path()))

            except Exception as e:
                logger.exception(e)
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
    def post(self, request, book, type, format=None):
        import json, zipfile, io, os
        pages = json.loads(request.body, encoding='utf-8').get('pages', [])
        book = Book(book)
        pages = book.pages() if len(pages) == 0 else [book.page(p) for p in pages]
        if type == 'annotations.zip':
            s = io.BytesIO()
            zf = zipfile.ZipFile(s, 'w')
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
        elif type == 'monodi2.zip':
            from omr.datatypes.monodi2_exporter import pcgts_to_monodi, PcGts
            pcgts = [PcGts.from_file(f) for f in [p.file('pcgts', False) for p in pages] if f.exists()]
            obj = pcgts_to_monodi(pcgts).to_json()

            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w') as zf:
                with zf.open(book.book + '.json', 'w') as f:
                    f.write(json.dumps(obj).encode('utf-8'))

            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.monodi2.zip')

        return Response(status=status.HTTP_400_BAD_REQUEST)

