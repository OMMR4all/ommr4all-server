from rest_framework.views import APIView
from rest_framework.response import Response
import rest_framework.status as status
from database import DatabaseBook
from database.database_permissions import DatabaseBookPermissionFlag
from restapi.models.auth import RestAPIUser
import logging
logger = logging.getLogger(__name__)


class UserBookPermissionsView(APIView):
    def get(self, request, book):
        book = DatabaseBook(book)
        user_permissions = book.resolve_user_permissions(request.user)
        if not user_permissions or user_permissions.flags == DatabaseBookPermissionFlag.NONE:
            raise Response(status=status.HTTP_404_NOT_FOUND)

        return Response(user_permissions.to_json())



class UserSelfView(APIView):
    """Identity of the requesting user.

    Fallback for clients whose stored session predates the username in the login
    response: a token refresh only replaces 'access', so they would never learn it."""
    def get(self, request):
        return Response(RestAPIUser.from_user(request.user).to_dict())
