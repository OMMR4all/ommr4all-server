from rest_framework.views import APIView
from rest_framework.response import Response
import rest_framework.status as status
from database import DatabaseBook
from database.database_permissions import DatabaseBookPermissionFlag
import logging
logger = logging.getLogger(__name__)


class UserBookPermissionsView(APIView):
    def get(self, request, book):
        book = DatabaseBook(book)
        user_permissions = book.resolve_user_permissions(request.user)
        if not user_permissions or user_permissions.flags == DatabaseBookPermissionFlag.NONE:
            raise Response(status=status.HTTP_404_NOT_FOUND)

        return Response(user_permissions.to_json())

