from typing import Union, Iterable
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from rest_framework.authtoken.models import Token

from django.contrib.auth.models import User, Group

from database.models.permissions import DatabasePermissionFlag
from .error import APIError, ErrorCodes


class require_global_permissions(object):
    def __init__(self, flags: Union[DatabasePermissionFlag, Iterable[DatabasePermissionFlag]]):
        if isinstance(flags, DatabasePermissionFlag):
            self.flags = ('database.' + flags.value, )
        else:
            self.flags = tuple('database.' + f.value for f in flags)

    def __call__(self, func):
        def wrapper_require_permissions(view, request, *args, **kwargs):
            if request.user.has_perms(self.flags):
                return func(view, request, *args, **kwargs)
            else:
                return APIError(status=status.HTTP_401_UNAUTHORIZED,
                                developerMessage='User {} has insufficient rights. Requested flags {}.'.format(
                                    request.user.username, self.flags),
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
