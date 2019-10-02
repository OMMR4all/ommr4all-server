from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.http import FileResponse
from database import *
from database.database_permissions import BookPermissionFlags
from database.models.permissions import DatabasePermissionFlag
from restapi.models.auth import RestAPIUser
from restapi.models.error import APIError, ErrorCodes
import json
import logging
import re
import os
from typing import List
logger = logging.getLogger(__name__)


class require_permissions(object):
    def __init__(self, flags: List[DatabaseBookPermissionFlag]):
        self.flags = flags

    def __call__(self, func):
        def wrapper_require_permissions(view, request, book, *args, **kwargs):
            book = DatabaseBook(book)
            user_permissions = book.resolve_user_permissions(request.user)
            if all([user_permissions.has(f) for f in self.flags]):
                return func(view, request, book.book, *args, **kwargs)
            else:
                return APIError(status=status.HTTP_401_UNAUTHORIZED,
                                developerMessage='User {} has insufficient rights on book {}. Requested flags {} on {}.'.format(
                                    request.user.username, book.book, self.flags, user_permissions),
                                userMessage='Insufficient permissions to access book {}'.format(book.book),
                                errorCode=ErrorCodes.BOOK_INSUFFICIENT_RIGHTS,
                                ).response()

        return wrapper_require_permissions


class BookMetaView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)

        return Response({**book.get_meta().to_dict(), 'permissions': book.resolve_user_permissions(request.user).flags})

    @require_permissions([DatabaseBookPermissionFlag.EDIT_BOOK_META])
    def put(self, request, book):
        from database.database_book_meta import DatabaseBookMeta
        book = DatabaseBook(book)
        meta = DatabaseBookMeta.from_book_json(book, request.body)
        book.save_json_to_meta(meta.to_dict())
        return Response()


class BookView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)
        pages = book.pages()

        pageIndex = int(request.query_params.get("pageIndex", 0))
        pageSize = int(request.query_params.get("pageSize", len(pages)))
        offset = pageIndex * pageSize

        paginated_pages = pages[offset:offset + pageSize]
        return Response({
            'totalPages': len(pages),
            'pages': sorted([{'label': page.page} for page in paginated_pages if page.is_valid()], key=lambda v: v['label'])})

    @require_permissions([DatabaseBookPermissionFlag.DELETE_BOOK])
    def delete(self, request, book, format=None):
        book = DatabaseBook(book)
        book.delete()
        return Response()


class BookUploadView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.ADD_PAGES])
    def post(self, request, book, format=None):
        from PIL import Image
        book = DatabaseBook(book)
        if not os.path.exists(book.local_path()):
            os.mkdir(book.local_path())

        for type, file in request.FILES.items():
            logger.debug('Received new image of content type {}'.format(file.content_type))
            name = os.path.splitext(os.path.basename(file.name))[0]
            name = re.sub('[^\w]', '_', name)
            type = file.content_type

            def image_to_page(img, page_name):
                page = DatabasePage(book, page_name)
                if not os.path.exists(page.local_path()):
                    os.mkdir(page.local_path())

                original = DatabaseFile(page, 'color_original')
                img.save(original.local_path())
                logger.debug('Created page at {}'.format(page.local_path()))

            try:
                if type.startswith('image/'):
                    img = Image.open(file.file, 'r').convert('RGB')
                    image_to_page(img, name)

                elif type == 'application/pdf':
                    from pdf2image import convert_from_bytes
                    images = convert_from_bytes(file.file.read())
                    for i, image in enumerate(images):
                        image_to_page(image, "{}_{:04d}".format(name, i))
                else:
                    return Response(status=status.HTTP_400_BAD_REQUEST)

            except Exception as e:
                logger.exception(e)
                return Response(status=status.HTTP_400_BAD_REQUEST)

        return Response()


class BooksImportView(APIView):
    # allow get access to any user, but prevent put
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def post(self, request):
        import zipfile
        for type, file in request.FILES.items():
            logger.debug('Received import of content type {}'.format(file.content_type))
            try:
                zf = zipfile.ZipFile(file.file, 'r')
                files = zf.filelist
                if len(files) == 0:
                    return

                base_dir = files[0].filename.split(os.sep)[0]
                if not all([f.filename.startswith(base_dir + os.sep) for f in files]):
                    return APIError(status=status.HTTP_400_BAD_REQUEST,
                                    developerMessage='Invalid zip file, not all names lie in the same subdir "{}"'.format(base_dir),
                                    userMessage='Invalid zip file. All files must be in a directory named "{}"'.format(base_dir),
                                    errorCode=ErrorCodes.BOOK_IMPORT_FAILED_INVALID_STRUCTURE,
                                    ).response()

                book = DatabaseBook(base_dir)
                if book.exists():
                    return APIError(status=status.HTTP_400_BAD_REQUEST,
                                    developerMessage='Book at {} with name already exists'.format(book, book.get_meta().name),
                                    userMessage='Book {} already exists'.format(book.get_meta().name),
                                    errorCode=ErrorCodes.BOOK_IMPORT_FAILED_BOOK_EXISTS,
                                    ).response()

                logger.info("Extracting imported file to {}".format(book.local_path(os.pardir)))
                zf.extractall(book.local_path(os.pardir))

            except Exception as e:
                logger.exception(e)
                return Response(status=status.HTTP_400_BAD_REQUEST)
        return Response()


class BooksView(APIView):
    # allow get access to any user, but prevent put
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def put(self, request, format=None):
        from database.database_book_meta import DatabaseBookMeta
        import datetime
        meta = DatabaseBookMeta.from_json(request.body, dict_params={'use_datetime': True})

        if not meta.name or len(meta.name) == 0:
            return APIError(status.HTTP_400_BAD_REQUEST,
                            "Received an empty book name",
                            "No book name provided",
                            ErrorCodes.BOOK_INVALID_NAME,
                            ).response()

        book_id = re.sub(r'[^\w]', '_', meta.name)

        try:
            b = DatabaseBook(book_id)
            if b.exists():
                return APIError(status.HTTP_409_CONFLICT,
                                "A book with the id {} already exists (requested name {})".format(book_id, meta.name),
                                "A book with the name {} already exists".format(meta.name),
                                ErrorCodes.BOOK_EXISTS
                                ).response()

            meta.id = b.book
            meta.creator = RestAPIUser.from_user(request.user)
            meta.created = datetime.datetime.now()

            if b.create(meta):
                # creator is administrator of book
                b.get_or_add_user_permissions(request.user, BookPermissionFlags.full_access_flags())
                b.get_permissions().write()
                return Response(b.get_meta().to_json())
            else:
                raise InvalidFileNameException(book_id)
        except InvalidFileNameException as e:
            logging.exception(e)
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Invalid filename for book (id={}, name={})".format(book_id, meta.name),
                            "Invalid book name: {}".format(meta.name),
                            ErrorCodes.BOOK_INVALID_NAME,
                            ).response()

    def get(self, request, format=None):
        # TODO: sort by in request
        books = DatabaseBook.list_available_book_metas_for_user(request.user, DatabaseBookPermissionFlag.READ)

        pageIndex = request.query_params.get("pageIndex", 0)
        pageSize = request.query_params.get("pageSize", len(books))  # by default all books

        paginatedBooks = books[pageIndex:pageIndex + pageSize]

        return Response({
            'totalPages': len(books),
            'books': sorted([{
                **book.to_dict(), **{'permissions': DatabaseBook(book.id).resolve_user_permissions(request.user).flags}
        } for book in paginatedBooks], key=lambda b: b['name'])})


class BookDownloaderView(APIView):
    permission_classes = [permissions.AllowAny]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, type):
        import json, zipfile, io, os
        pages = json.loads(request.body, encoding='utf-8').get('pages', [])
        book = DatabaseBook(book)
        pages = book.pages() if len(pages) == 0 else [book.page(p) for p in pages]
        if type == 'annotations.zip':
            s = io.BytesIO()
            zf = zipfile.ZipFile(s, 'w')
            for page in pages:
                file_names = ['color_original', 'color_norm_x2', 'binary_norm_x2', 'pcgts', 'meta']
                files = [page.file(f) for f in file_names]

                if any([not f.exists() for f in files]):
                    continue

                for file, fn in zip(files, file_names):
                    zf.write(file.local_path(), os.path.join(fn, page.page + file.ext()))

            zf.close()
            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.zip')
        elif type == 'backup.zip':
            s = io.BytesIO()
            zf = zipfile.ZipFile(s, 'w')
            files_to_ignore = [re.compile(r".*\.zip$")]
            for root, dirs, files in os.walk(book.local_path()):
                for file in files:
                    if any([f.match(file) for f in files_to_ignore]):
                        continue

                    f = os.path.join(root, file)
                    zf.write(f, os.path.join(book.book, os.path.relpath(f, book.local_path())))

            zf.close()
            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.backup.zip')
        elif type == 'monodiplus.json':
            from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
            from database.file_formats import PcGts
            pcgts = [PcGts.from_file(f) for f in [p.file('pcgts', False) for p in pages] if f.exists()]
            obj = PcgtsToMonodiConverter(pcgts).root.to_json()

            s = io.BytesIO()
            s.write(json.dumps(obj, indent=2).encode('utf-8'))
            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.json')
        elif type == 'monodiplus.zip':
            from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
            from database.file_formats import PcGts
            pcgts = [PcGts.from_file(f) for f in [p.file('pcgts', False) for p in pages] if f.exists()]
            obj = PcgtsToMonodiConverter(pcgts).root.to_json()

            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w') as zf:
                with zf.open(book.book + '.json', 'w') as f:
                    f.write(json.dumps(obj, indent=2).encode('utf-8'))

            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.monodi2.zip')
        elif type == 'mei4.zip':
            from database.file_formats.exporter.mei.pcgts_to_mei4_exporter import PcgtsToMeiConverter
            from database.file_formats import PcGts
            pcgts = [PcGts.from_file(f) for f in [p.file('pcgts', False) for p in pages] if f.exists()]

            s = io.BytesIO()
            with zipfile.ZipFile(s, 'w') as zf:
                for p in pcgts:
                    with zf.open(os.path.join(book.book, p.page.location.page + '.xml'), 'w') as f:
                        PcgtsToMeiConverter(p).write(f)

            s.seek(0)
            return FileResponse(s, as_attachment=True, filename=book.book + '.mei.zip')

        return Response(status=status.HTTP_400_BAD_REQUEST)


class BookRenamePagesView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.RENAME_PAGES])
    def post(self, request, book):
        book = DatabaseBook(book)
        body = json.loads(request.body)
        page_names = set(book.page_names())
        files = body.get('files', [])
        pairs = [(file['src'], file['target']) for file in files]
        srcs, targets = zip(*pairs)

        # checks for uniqueness
        if len(set(srcs)) != len(srcs):
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Source files are not unique: {}".format(srcs),
                            "Files are not unique",
                            ErrorCodes.BOOK_PAGES_RENAME_REQUIRE_UNIQUE_SOURCES
                            ).response()

        if len(set(targets)) != len(targets):
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Target files are not unique: {}".format(targets),
                            "Files are not unique",
                            ErrorCodes.BOOK_PAGES_RENAME_REQUIRE_UNIQUE_TARGETS
                            ).response()

        # check that no target exists
        intersection = set(targets).intersection(page_names.difference(set(srcs)))
        if len(intersection) != 0:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Target filename already exists: {}".format(intersection),
                            "Target page(s) {} already exist(s).".format(", ".join(list(intersection))),
                            ErrorCodes.BOOK_PAGES_RENAME_TARGET_EXISTS,
                            ).response()

        try:
            # create prefix for temporary files
            tmp_prefix = '_'
            while len({tmp_prefix + s for s in srcs}.intersection(page_names)) != 0:
                tmp_prefix += '_'

            pages = [book.page(p) for p in srcs]

            # move to temporary files
            for page in pages:
                page.rename(tmp_prefix + page.page)

            # move to true targets
            for page, target in zip(pages, targets):
                page.rename(target)

        except InvalidFileNameException as e:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Renaming page not possible, because the new name '{}' is invalid.".format(e.filename),
                            "Invalid page name '{}'".format(e.filename),
                            ErrorCodes.PAGE_INVALID_NAME,
                            ).response()
        except FileExistsException as e:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Renaming page not possible, because a file at '{}' already exists".format(e.filename),
                            "A file at '{}' already exists".format(e.filename),
                            ErrorCodes.PAGE_EXISTS,
                            ).response()

        return Response()
