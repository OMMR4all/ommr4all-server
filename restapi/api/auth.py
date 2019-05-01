from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from django.contrib.auth.models import User, Group


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
