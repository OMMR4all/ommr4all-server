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


def is_admin(user, flag: DatabasePermissionFlag) -> bool:
    """Whether ``user`` may use an administrative feature.

    Django's staff/superuser flags always qualify; in addition the feature's own global permission
    can be granted to a user or a group, so administrative rights can be handed out selectively
    without giving access to the Django admin.
    """
    if not user or not user.is_authenticated or not user.is_active:
        return False
    return bool(user.is_superuser or user.is_staff or user.has_perm('database.' + flag.value))


class require_admin(object):
    """Like require_global_permissions, but also accepts Django's is_staff/is_superuser."""
    def __init__(self, flag: DatabasePermissionFlag):
        self.flag = flag

    def __call__(self, func):
        def wrapper_require_admin(view, request, *args, **kwargs):
            if is_admin(request.user, self.flag):
                return func(view, request, *args, **kwargs)
            return APIError(status=status.HTTP_401_UNAUTHORIZED,
                            developerMessage='User {} has insufficient rights. Requires admin or {}.'.format(
                                request.user.username, 'database.' + self.flag.value),
                            userMessage='Insufficient permissions',
                            errorCode=ErrorCodes.GLOBAL_INSUFFICIENT_RIGHTS,
                            ).response()

        return wrapper_require_admin


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
