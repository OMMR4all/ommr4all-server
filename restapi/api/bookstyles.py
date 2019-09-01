from rest_framework.views import APIView
from rest_framework import permissions
from .auth import require_global_permissions, DatabasePermissionFlag
from .error import APIError, ErrorCodes
from rest_framework import status
from rest_framework.response import Response
from database.models.bookstyles import BookStyle, BookStyleSerializer
import logging
logger = logging.getLogger(__name__)


class BookStylesView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_global_permissions(DatabasePermissionFlag.ADD_BOOK_STYLE)
    def put(self, request):
        serializer = BookStyleSerializer(request.body)
        if not serializer.is_valid():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request body: {}".format(request.body),
                userMessage='Invalid request',
                errorCode=ErrorCodes.BOOK_STYLE_INVALID_REQUEST,
            ).response()
        serializer.save()
        return Response()

    def get(self, request):
        return Response(BookStyleSerializer(BookStyle.objects.all(), many=True).data)


class BookStyleView(APIView):
    @require_global_permissions(DatabasePermissionFlag.DELETE_BOOK_STYLE)
    def delete(self, request, id):
        BookStyle.objects.get(id=id).delete()
        return Response()

    @require_global_permissions(DatabasePermissionFlag.EDIT_BOOK_STYLE)
    def post(self, request, id):
        serializer = BookStyleSerializer(BookStyle.objects.get(id=id), request.body)
        serializer.save()
        return Response()
