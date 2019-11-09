from rest_framework.views import APIView
from rest_framework import status, permissions
from database import *
from database.file_formats.performance.pageprogress import PageProgress
from database.file_formats.performance.statistics import Statistics
from database.file_formats.pcgts import PcGts
from restapi.views.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.models.error import *
from django.views.static import serve
from json import JSONDecodeError
import logging
import json
import re
import zipfile
import datetime

logger = logging.getLogger(__name__)


def require_lock(func):
    def wrapper(view, request, book, page, *args, **kwargs):
        page = DatabaseBook(book).page(page)
        if not page.is_locked_by_user(request.user):
            return PageNotLockedAPIError(status.HTTP_423_LOCKED).response()
        else:
            return func(view, request, book, page.page, *args, **kwargs)

    return wrapper


class require_page_verification(object):
    def __init__(self, verified = True):
        self.verified = verified

    def __call__(self, func):
        def wrapper_require_permissions(view, request, book, page, *args, **kwargs):
            book = DatabaseBook(book)
            page = book.page(page)
            pp = page.page_progress()
            if pp.verified == self.verified:
                return func(view, request, book.book, page.page, *args, **kwargs)
            else:
                return APIError(status=status.HTTP_403_FORBIDDEN,
                                developerMessage='Page ({}/{}) verification mismatch. Required {} but got {}.'.format(
                                    book.book, page.page, self.verified, pp.verified),
                                userMessage='Verification mismatch in page {} of book {}'.format(
                                    page.page, book.book),
                                errorCode=ErrorCodes.PAGE_VERIFICATION_REQUIRED if self.verified else ErrorCodes.PAGE_IS_VERIFIED,
                                ).response()

        return wrapper_require_permissions


class PageContentView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page, content):
        book = DatabaseBook(book)
        page = DatabasePage(book, page)
        file = DatabaseFile(page, content)

        if not file.exists():
            file.create()

        return serve(request._request, file.local_request_path(), "/", False)


class PageLockView(APIView):
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        return Response({'locked': page.is_locked_by_user(request.user)})

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    @require_page_verification(False)
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
        return Response(pp.verified)

    @require_permissions([DatabaseBookPermissionFlag.VERIFY_PAGE])
    def put(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        pp = page.page_progress()
        if pp.verified_allowed():
            pp.verified = True
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
        pp.verified = False
        page.save_page_progress()
        return Response()


class PageProgressView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    @require_lock
    def put(self, request, book, page):
        book = DatabaseBook(book)
        page = DatabasePage(book, page)

        obj = json.loads(request.body, encoding='utf-8')
        pp = page.page_progress()
        user_permissions = book.resolve_user_permissions(request.user)
        verify_allowed = user_permissions.has(DatabaseBookPermissionFlag.VERIFY_PAGE)
        pp.merge_local(PageProgress.from_dict(obj), locks=True, verified=verify_allowed)
        page.set_page_progress(pp)
        page.save_page_progress()

        # add to backup archive
        with zipfile.ZipFile(page.file('page_progress_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('page_progress_{}.json'.format(datetime.datetime.now()), json.dumps(pp.to_json(), indent=2))

        return Response()

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        page = DatabasePage(DatabaseBook(book), page)
        return Response(page.page_progress().to_dict())


class PagePcGtsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    @require_lock
    def put(self, request, book, page):
        book = DatabaseBook(book)
        page = DatabasePage(book, page)
        obj = json.loads(request.body, encoding='utf-8')

        pcgts = PcGts.from_json(obj, page)
        pcgts.to_file(page.file('pcgts').local_path())

        # add to backup archive
        with zipfile.ZipFile(page.file('pcgts_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('pcgts_{}.json'.format(datetime.datetime.now()), json.dumps(pcgts.to_json(), indent=2))

        logger.debug('Successfully saved pcgts file to {}'.format(page.file('pcgts').local_path()))

        return Response()

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
    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    @require_lock
    def put(self, request, book, page):
        book = DatabaseBook(book)
        page = DatabasePage(book, page)

        obj = json.loads(request.body, encoding='utf-8')
        page.set_page_statistics(Statistics.from_json(obj))
        page.save_page_statistics()

        # add to backup archive
        with zipfile.ZipFile(page.file('statistics_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('statistics_{}.json'.format(datetime.datetime.now()), json.dumps(page.page_statistics().to_json(), indent=2))

        logger.debug('Successfully saved statistics file to {}'.format(page.file('statistics').local_path()))

        return Response()

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        return Response(DatabaseBook(book).page(page).page_statistics().to_json())


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
