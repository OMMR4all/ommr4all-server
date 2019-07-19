from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from django.contrib.auth.models import User, Group

from database.database_permissions import DatabasePermissionFlag
from .error import APIError, ErrorCodes


class require_global_permissions(object):
    def __init__(self, flags: DatabasePermissionFlag):
        self.flags = flags

    def __call__(self, func):
        def wrapper_require_permissions(view, request, book, *args, **kwargs):
            # TODO: load actual permissions
            user_permissions = DatabasePermissionFlag.ADD_BOOK_STYLE | DatabasePermissionFlag.DELETE_BOOK_STYLE | DatabasePermissionFlag.EDIT_BOOK_STYLE
            if (user_permissions & self.flags) == self.flags:
                return func(view, request, book.book, *args, **kwargs)
            else:
                return APIError(status=status.HTTP_401_UNAUTHORIZED,
                                developerMessage='User {} has insufficient rights. Requested flags {} on {}.'.format(
                                    request.user.username, self.flags, user_permissions),
                                userMessage='Insufficient permissions',
                                errorCode=ErrorCodes.GLOBAL_INSUFFICIENT_RIGHTS,
                                ).response()

        return wrapper_require_permissions


class AuthView(APIView):
    def get(self, request, auth):
        if auth == 'users':
            users = User.objects.all()
            return Response(
                {'users': [{
                    'username': u.username,
                    'firstName': u.first_name,
                    'lastName': u.last_name
                } for u in users]})
        elif auth == 'groups':
            groups = Group.objects.all()
            return Response(
                {'groups': [{
                    'name': g.name
                } for g in groups]})
        else:
            return Response(status.HTTP_404_NOT_FOUND)
