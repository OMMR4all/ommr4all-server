from dataclasses import dataclass

from dataclasses_json import dataclass_json
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
import urllib.parse
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


class BookStatsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)
        from database.tools.book_statistics import Counts
        from database.book_index import book_counts

        @dataclass_json
        @dataclass
        class DatasetStatisticsResult:
            current: int
            total: int
            counts: Counts

        return Response(DatasetStatisticsResult(0, 0, book_counts(book)).to_dict())


def etag_response(request, payload) -> Response:
    """Response with a content-derived ETag; 304 without a body when the client's
    If-None-Match still matches. The stats are recomputed either way (they are
    cheap after the index sync), but polling clients skip transfer and re-render."""
    import hashlib
    etag = '"{}"'.format(hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32])
    if_none_match = [t.strip() for t in request.headers.get('If-None-Match', '').split(',') if t.strip()]
    if etag in if_none_match:
        return Response(status=status.HTTP_304_NOT_MODIFIED, headers={'ETag': etag})
    return Response(payload, headers={'ETag': etag})


class BookOverviewStatsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        from database.tools.book_overview_stats import compute_overview_stats
        return etag_response(request, compute_overview_stats(DatabaseBook(book)))


class BooksOverviewStatsView(APIView):
    """Aggregated overview stats of every readable book in a single request
    (the client's books table needs one call instead of one per book)."""
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request):
        from database.book_index import list_books_synced
        from database.tools.book_overview_stats import compute_overview_stats
        stats = {}
        for row in list_books_synced():
            db_book = DatabaseBook(row.name)
            if db_book.resolve_user_permissions(request.user).has(DatabaseBookPermissionFlag.READ):
                # row is already meta-synced -- don't make compute re-resolve it per book
                stats[row.name] = compute_overview_stats(db_book, book_row=row)
        return etag_response(request, stats)


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
        stored = None
        if meta.updated is None:
            # clients do not send the last-modified timestamp; keep the stored one
            stored = book.get_meta()
            meta.updated = stored.updated
            meta.updatedBy = stored.updatedBy
        # a client that does not know the operation locks must not unlock the book by
        # putting a meta without them (they are absent, not False)
        if meta.lockBookOperations is None or meta.lockTraining is None:
            stored = stored if stored is not None else book.get_meta()
            if meta.lockBookOperations is None:
                meta.lockBookOperations = stored.lockBookOperations
            if meta.lockTraining is None:
                meta.lockTraining = stored.lockTraining
        book.save_json_to_meta(meta.to_dict())
        return Response()


class BookView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)
        pages = book.pages()

        # filtering before paginating keeps totalPages in sync with what the client can page
        # through -- filtering the current window only would hide most of the matches
        name_filter = request.query_params.get("filter", "").strip().lower()
        if name_filter:
            pages = [page for page in pages if name_filter in page.page.lower()]

        pageIndex = int(request.query_params.get("pageIndex", 0))
        pageSize = int(request.query_params.get("pageSize", len(pages)))
        offset = pageIndex * pageSize

        # book.pages() already filters to page folders and returns them sorted
        paginated_pages = pages[offset:offset + pageSize]
        return Response({
            'totalPages': len(pages),
            'pages': [{'label': page.page} for page in paginated_pages]})

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
        # names of the pages actually created, reported back so the client can show a
        # page count instead of a file count (a single PDF becomes many pages)
        created_pages = []
        for type, file in request.FILES.items():
            logger.debug('Received new image of content type {}'.format(file.content_type))
            name = os.path.splitext(os.path.basename(file.name))[0]
            name = re.sub(r'[^\w]', '_', name)
            type = file.content_type

            def image_to_page(img, page_name):
                page = DatabasePage(book, page_name)
                if not os.path.exists(page.local_path()):
                    os.mkdir(page.local_path())

                original = DatabaseFile(page, 'color_original')
                img.save(original.local_path())
                logger.debug('Created page at {}'.format(page.local_path()))
                page.mark_updated(request.user, propagate=False)
                created_pages.append(page.page)

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

        book.mark_updated(request.user)
        return Response({'pages': created_pages})


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

                from database.book_index import safe_index_book, safe_index_documents
                safe_index_book(book, force=True)
                safe_index_documents(book)

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
        meta = DatabaseBookMeta.from_json(request.body)

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
        from database.book_index import list_books_for_user
        books = list_books_for_user(request.user, DatabaseBookPermissionFlag.READ)
        pageIndex = int(request.query_params.get("pageIndex", 0))
        pageSize = int(request.query_params.get("pageSize", len(books)))  # by default all books

        paginatedBooks = books[pageIndex * pageSize:pageIndex * pageSize + pageSize] if pageSize else books
        return Response({
            'totalPages': len(books),
            'books': sorted([{
                **meta, **{'permissions': permission_flags}
        } for meta, permission_flags in paginatedBooks], key=lambda b: b['name'])})


DOWNLOAD_TOKEN_SALT = 'ommr4all.book.download'
# the token travels in a query string and therefore ends up in the access log,
# so it is deliberately only valid long enough to start the download
DOWNLOAD_TOKEN_MAX_AGE = 600


class BookDownloadTokenView(APIView):
    """Hands out a short lived, signed URL for a book download.

    The client authenticates with a JWT in the Authorization header, which a plain
    browser navigation cannot send. Letting the browser fetch the archive itself is
    what gives the user a real progress bar with speed and ETA (and keeps a multi
    gigabyte backup out of the tab's memory), so it needs a URL that carries its own
    proof of access.
    """
    permission_classes = [permissions.AllowAny]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, type):
        from django.core import signing
        body = json.loads(request.body) if request.body else {}
        pages = body.get('pages', [])
        full = bool(body.get('full', False))
        book = DatabaseBook(book)

        # the dialog always sends every page name; a few hundred of them in a query
        # string would run into Apache's LimitRequestLine, and "all pages" is the
        # default anyway
        if len(pages) > 0 and set(pages) == set(book.page_names()):
            pages = []

        token = signing.dumps({'u': request.user.pk, 'b': book.book, 't': type, 'p': pages, 'f': full},
                              salt=DOWNLOAD_TOKEN_SALT)
        # relative on purpose: the browser then downloads from the origin it is already on,
        # which works through the dev proxy and behind a TLS terminator without having to
        # trust any forwarded host header
        return Response({
            'url': '{}?token={}'.format(request.path[:-len('/token')], urllib.parse.quote(token)),
            'filename': download_filename(book, type),
        })


class BookDownloaderView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, book, type):
        """Token authenticated download, used for native browser downloads.

        Deliberately *not* decorated with require_permissions: this view is AllowAny
        and resolve_user_permissions grants an anonymous user the book's default
        flags, so a decorated GET would serve any publicly readable book without a
        token at all. The signature is the only gate here.
        """
        from django.core import signing
        from django.contrib.auth.models import User

        try:
            payload = signing.loads(request.query_params.get('token', ''),
                                    salt=DOWNLOAD_TOKEN_SALT, max_age=DOWNLOAD_TOKEN_MAX_AGE)
        except signing.BadSignature:
            return APIError(status=status.HTTP_403_FORBIDDEN,
                            developerMessage='Invalid or expired download token for book {}'.format(book),
                            userMessage='This download link is invalid or has expired. Please start the download again.',
                            errorCode=ErrorCodes.BOOK_INSUFFICIENT_RIGHTS,
                            ).response()

        if payload.get('b') != book or payload.get('t') != type:
            return APIError(status=status.HTTP_403_FORBIDDEN,
                            developerMessage='Download token for {}/{} used on {}/{}'.format(
                                payload.get('b'), payload.get('t'), book, type),
                            userMessage='This download link is invalid.',
                            errorCode=ErrorCodes.BOOK_INSUFFICIENT_RIGHTS,
                            ).response()

        # permissions are re-resolved instead of trusted from the token, so access
        # revoked after the token was issued takes effect immediately
        user = User.objects.filter(pk=payload.get('u')).first()
        db_book = DatabaseBook(book)
        if user is None or not db_book.resolve_user_permissions(user).has(DatabaseBookPermissionFlag.READ):
            return APIError(status=status.HTTP_401_UNAUTHORIZED,
                            developerMessage='User {} has insufficient rights on book {}'.format(payload.get('u'), book),
                            userMessage='Insufficient permissions to access book {}'.format(book),
                            errorCode=ErrorCodes.BOOK_INSUFFICIENT_RIGHTS,
                            ).response()

        return build_download_response(db_book, type, payload.get('p', []), payload.get('f', False))

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def post(self, request, book, type):
        body = json.loads(request.body) if request.body else {}
        return build_download_response(DatabaseBook(book), type,
                                       body.get('pages', []), bool(body.get('full', False)))


def download_filename(book: DatabaseBook, type: str) -> str:
    """`<book title>.<type>`, e.g. "Graduel de Nevers.backup.zip"."""
    from restapi.views.downloads import sanitize_filename
    return '{}.{}'.format(sanitize_filename(book.get_meta().name), type)


def build_download_response(book: DatabaseBook, type: str, page_names: List[str], full: bool = False):
    import zipfile, io
    from restapi.views.downloads import (annotation_entries, backup_entries, original_image_entries,
                                         stream_zip_response)
    pages = book.pages() if len(page_names) == 0 else [book.page(p) for p in page_names]
    filename = download_filename(book, type)

    # the file based exports are streamed from disk: they carry whole images and
    # would otherwise be held in the wsgi process in full before anything is sent
    if type == 'annotations.zip':
        return stream_zip_response(annotation_entries(pages), filename)
    elif type == 'backup.zip':
        # a backup is always the whole book, the page selection does not apply
        return stream_zip_response(backup_entries(book, full=full), filename)
    elif type == 'original_images.zip':
        return stream_zip_response(original_image_entries(book, pages), filename)
    # the remaining exports are generated from the pcgts and are small enough to build in memory
    elif type == 'monodiplus.json':
        from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
        from database.file_formats import PcGts
        pcgts = [PcGts.from_file(f) for f in [p.file('pcgts', False) for p in pages] if f.exists()]
        obj = PcgtsToMonodiConverter(pcgts).root.to_json()

        s = io.BytesIO()
        s.write(json.dumps(obj, indent=2).encode('utf-8'))
        s.seek(0)
        return FileResponse(s, as_attachment=True, filename=filename)
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
        return FileResponse(s, as_attachment=True, filename=filename)
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
        return FileResponse(s, as_attachment=True, filename=filename)

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

        book.mark_updated(request.user)
        return Response()
