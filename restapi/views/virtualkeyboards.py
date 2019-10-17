from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from database import DatabaseBook
from restapi.views.bookaccess import require_permissions, DatabaseBookPermissionFlag

import os
import json


class BookVirtualKeyboardView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, book):
        book = DatabaseBook(book)

        if not book.exists():
            return Response(status=status.HTTP_400_BAD_REQUEST)

        file = book.local_path("virtual_keyboard.json")
        if not os.path.exists(file):
            file = book.local_default_virtual_keyboards_path('default.json')

        with open(file) as f:
            return Response(json.load(f))

    @require_permissions([DatabaseBookPermissionFlag.READ_WRITE])
    def put(self, request, book):
        book = DatabaseBook(book)

        if not book.exists():
            return Response(status=status.HTTP_400_BAD_REQUEST)

        file = book.local_path("virtual_keyboard.json")
        json.dump(json.loads(request.body, encoding="utf-8"), open(file, 'w'), indent=4)
        return Response()
