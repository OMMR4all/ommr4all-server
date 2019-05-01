from rest_framework.views import APIView
from restapi.api.bookaccess import require_permissions
from restapi.api.error import ErrorCodes, APIError
from rest_framework.response import Response
import rest_framework.status as status
from database import DatabaseBook
from database.database_permissions import BookPermissionFlags, DatabaseBookPermissionFlag
import logging
import json
logger = logging.getLogger(__name__)


class UserBookPermissionsView(APIView):
    def get(self, request, book):
        book = DatabaseBook(book)
        user_permissions = book.resolve_user_permissions(request.user)
        if not user_permissions or user_permissions.flags == DatabaseBookPermissionFlag.NONE:
            raise Response(status=status.HTTP_404_NOT_FOUND)

        return Response(user_permissions.to_json())

