from rest_framework.views import APIView
from restapi.views.bookaccess import require_permissions, DatabaseBookPermissionFlag
from restapi.models.error import ErrorCodes, APIError
from rest_framework.response import Response
import rest_framework.status as status
from database import DatabaseBook
from database.database_permissions import BookPermissionFlags, DatabaseBookPermissionFlag
import logging
import json
logger = logging.getLogger(__name__)


class BookDefaultPermissionsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def post(self, request, book):
        flags = json.loads(request.body)['flags']
        flags = flags & DatabaseBookPermissionFlag.ALLOWED_OTHER_PERMISSIONS
        book = DatabaseBook(book)
        permissions = book.get_permissions().permissions
        permissions.default.flags = flags
        book.get_permissions().write()
        return Response(permissions.default.to_json())


class BookUserPermissionsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def put(self, request, book, username):
        flags = json.loads(request.body).get('flags', DatabaseBookPermissionFlag.NONE)
        book = DatabaseBook(book)
        permissions = book.get_permissions().get_or_add_user_permissions(username, BookPermissionFlags(flags))
        return Response(permissions.to_json(), status.HTTP_201_CREATED)

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def delete(self, request, book, username):
        DatabaseBook(book).get_permissions().delete_user_permissions(username)
        return Response(status=status.HTTP_200_OK)

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def post(self, request, book, username):
        flags = json.loads(request.body)['flags']
        permissions = DatabaseBook(book).get_permissions().get_or_add_user_permissions(username, BookPermissionFlags(flags))
        return Response(permissions.to_json())


class BookGroupPermissionsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def put(self, request, book, name):
        flags = json.loads(request.body).get('flags', DatabaseBookPermissionFlag.NONE)
        permissions = DatabaseBook(book).get_permissions().get_or_add_group_permissions(name, BookPermissionFlags(flags))
        return Response(permissions.to_json(), status.HTTP_201_CREATED)

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def delete(self, request, book, name):
        DatabaseBook(book).get_permissions().delete_group_permissions(name)
        return Response(status=status.HTTP_200_OK)

    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def post(self, request, book, name):
        flags = json.loads(request.body)['flags']
        permissions = DatabaseBook(book).get_permissions().get_or_add_group_permissions(name, BookPermissionFlags(flags))
        return Response(permissions.to_json())


class BookPermissionsView(APIView):
    @require_permissions([DatabaseBookPermissionFlag.EDIT_PERMISSIONS])
    def get(self, request, book):
        book = DatabaseBook(book)
        return Response(book.get_permissions().permissions.to_json())
