from typing import Union, Iterable
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from django.contrib.auth.models import User, Group

from database.models.permissions import DatabasePermissionFlag
from restapi.models.error import APIError, ErrorCodes
from restapi.models.auth import RestAPIGroup, RestAPIUser


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
                {'users': [RestAPIUser.from_user(u).to_dict() for u in users]})
        elif auth == 'groups':
            groups = Group.objects.all()
            return Response(
                {'groups': [RestAPIGroup.from_group(g).to_dict() for g in groups]})
        else:
            return Response(status.HTTP_404_NOT_FOUND)
