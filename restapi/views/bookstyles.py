from rest_framework.views import APIView
from rest_framework import permissions
from .auth import require_global_permissions, DatabasePermissionFlag
from restapi.models.error import APIError, ErrorCodes
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
import re
from database.models.bookstyles import BookStyle, BookStyleSerializer
import logging
logger = logging.getLogger(__name__)


class BookStylesView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_global_permissions(DatabasePermissionFlag.ADD_BOOK_STYLE)
    def put(self, request):
        data = JSONParser().parse(request)
        if 'name' not in data or len(data['name']) == 0:
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid name in request body: data".format(data),
                userMessage='Invalid name. Got {}'.format(data.get('name'), None),
                errorCode=ErrorCodes.BOOK_STYLE_INVALID_NAME,
            ).response()

        data['id'] = re.sub(r'[^\w]', '_', data.get('name', ''))
        try:
            BookStyle.objects.get(id=data['id'])
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request to create a new book style. Id {} already existing".format(data['id']),
                userMessage='Book style with name {} ({}) already exists'.format(data['name'], data['id']),
                errorCode=ErrorCodes.BOOK_STYLE_EXISTS,
            ).response()
        except BookStyle.DoesNotExist:
            pass

        serializer = BookStyleSerializer(data=data)
        if not serializer.is_valid():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request body: {}".format(request.body),
                userMessage='Invalid request',
                errorCode=ErrorCodes.BOOK_STYLE_INVALID_REQUEST,
            ).response()
        serializer.save()
        return Response(serializer.data)

    def get(self, request):
        return Response(BookStyleSerializer(BookStyle.objects.all(), many=True).data)


class BookStyleView(APIView):
    @require_global_permissions(DatabasePermissionFlag.DELETE_BOOK_STYLE)
    def delete(self, request, id):
        BookStyle.objects.get(id=id).delete()
        return Response()

    @require_global_permissions(DatabasePermissionFlag.EDIT_BOOK_STYLE)
    def post(self, request, id):
        data = JSONParser().parse(request)
        serializer = BookStyleSerializer(BookStyle.objects.get(id=id), data=data)
        if not serializer.is_valid():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request body: {}".format(request.body),
                userMessage='Invalid request',
                errorCode=ErrorCodes.BOOK_STYLE_INVALID_REQUEST,
            ).response()
        serializer.save()
        return Response()
