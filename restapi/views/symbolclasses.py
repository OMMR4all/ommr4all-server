from rest_framework.views import APIView
from rest_framework import permissions
from .auth import require_global_permissions, DatabasePermissionFlag
from restapi.models.error import APIError, ErrorCodes
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from django.db import models
import re
from database.file_formats.pcgts.page.musicsymbol import NoteType, ClefType, AccidType
from database.models.symbolclasses import SymbolClass, SymbolClassSerializer, SYMBOL_CLASS_BASE_TYPES
import logging
logger = logging.getLogger(__name__)

BUILTIN_DIGIT_SHORTCUTS = {1, 2, 3, 4, 5, 6, 7}   # mirrors SYMBOL_CLASS_REGISTRY on the client


def digit_shortcut_conflict(digit, style_id, exclude_id=None) -> bool:
    if digit is None:
        return False
    if digit in BUILTIN_DIGIT_SHORTCUTS:
        return True
    q = SymbolClass.objects.filter(digit_shortcut=digit).exclude(id=exclude_id or '')
    if style_id is None:
        return q.exists()                      # a global class conflicts with every style
    return q.filter(models.Q(style_id=style_id) | models.Q(style__isnull=True)).exists()


def base_class_valid(base_symbol_type, base_sub_type) -> bool:
    if base_symbol_type not in SYMBOL_CLASS_BASE_TYPES:
        return False
    try:
        if base_symbol_type == 'note':
            NoteType(int(base_sub_type))
        elif base_symbol_type == 'clef':
            ClefType(base_sub_type)
        elif base_symbol_type == 'accid':
            AccidType(base_sub_type)
        else:
            # 'other' refines no built-in class, hence it must not claim a sub type
            return not base_sub_type
    except (ValueError, TypeError):
        return False
    return True


def validate_common(data, exclude_id=None):
    """Validation shared by create and edit. Returns an `APIError` response or None."""
    if not base_class_valid(data.get('base_symbol_type'), data.get('base_sub_type')):
        return APIError(
            status=status.HTTP_400_BAD_REQUEST,
            developerMessage="Invalid base class {}/{} in request body".format(
                data.get('base_symbol_type'), data.get('base_sub_type')),
            userMessage='Invalid base symbol class {} ({})'.format(
                data.get('base_symbol_type'), data.get('base_sub_type')),
            errorCode=ErrorCodes.SYMBOL_CLASS_INVALID_REQUEST,
        ).response()

    if digit_shortcut_conflict(data.get('digit_shortcut'), data.get('style'), exclude_id):
        return APIError(
            status=status.HTTP_400_BAD_REQUEST,
            developerMessage="Digit shortcut {} is already taken".format(data.get('digit_shortcut')),
            userMessage='The digit shortcut {} is already in use'.format(data.get('digit_shortcut')),
            errorCode=ErrorCodes.SYMBOL_CLASS_DIGIT_SHORTCUT_TAKEN,
        ).response()

    if data.get('base_symbol_type') == 'other' and not (data.get('glyph_preset') or data.get('svg_path')):
        return APIError(
            status=status.HTTP_400_BAD_REQUEST,
            developerMessage="Symbol class of base type 'other' without a glyph: {}".format(data),
            userMessage='A symbol class that is based on no built-in symbol needs its own glyph',
            errorCode=ErrorCodes.SYMBOL_CLASS_GLYPH_REQUIRED,
        ).response()
    return None


class SymbolClassesView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_global_permissions(DatabasePermissionFlag.ADD_SYMBOL_CLASS)
    def put(self, request):
        data = JSONParser().parse(request)
        if 'name' not in data or len(data['name']) == 0:
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid name in request body: {}".format(data),
                userMessage='Invalid name. Got {}'.format(data.get('name', None)),
                errorCode=ErrorCodes.SYMBOL_CLASS_INVALID_NAME,
            ).response()

        data['id'] = re.sub(r'[^\w]', '_', data.get('name', ''))
        if SymbolClass.objects.filter(id=data['id']).exists():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request to create a new symbol class. Id {} already existing".format(
                    data['id']),
                userMessage='Symbol class with name {} ({}) already exists'.format(data['name'], data['id']),
                errorCode=ErrorCodes.SYMBOL_CLASS_EXISTS,
            ).response()

        error = validate_common(data)
        if error is not None:
            return error

        serializer = SymbolClassSerializer(data=data)
        if not serializer.is_valid():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request body: {}".format(request.body),
                userMessage='Invalid request',
                errorCode=ErrorCodes.SYMBOL_CLASS_INVALID_REQUEST,
            ).response()
        serializer.save()
        return Response(serializer.data)

    def get(self, request):
        return Response(SymbolClassSerializer(SymbolClass.objects.all(), many=True).data)


class SymbolClassView(APIView):
    @require_global_permissions(DatabasePermissionFlag.DELETE_SYMBOL_CLASS)
    def delete(self, request, id):
        SymbolClass.objects.get(id=id).delete()
        return Response()

    @require_global_permissions(DatabasePermissionFlag.EDIT_SYMBOL_CLASS)
    def post(self, request, id):
        data = JSONParser().parse(request)
        data['id'] = id
        error = validate_common(data, exclude_id=id)
        if error is not None:
            return error

        serializer = SymbolClassSerializer(SymbolClass.objects.get(id=id), data=data)
        if not serializer.is_valid():
            return APIError(
                status=status.HTTP_400_BAD_REQUEST,
                developerMessage="Invalid request body: {}".format(request.body),
                userMessage='Invalid request',
                errorCode=ErrorCodes.SYMBOL_CLASS_INVALID_REQUEST,
            ).response()
        serializer.save()
        return Response()
