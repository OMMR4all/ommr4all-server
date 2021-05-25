import json
import os
import re
import subprocess

from database.file_formats.book import document
from database.file_formats.exporter.midi.simple_midi import SimpleMidiExporter

from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from database import *
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats import PcGts
import logging

from database.file_formats.book.document import Document
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
from database.file_formats.pcgts import PageScaleReference
from ommr4all.settings import BASE_DIR
from omr.util import PerformanceCounter
from restapi.models.error import APIError, ErrorCodes
from restapi.views.bookaccess import require_permissions
from restapi.views.pageaccess import require_lock
import requests as python_request

logger = logging.getLogger(__name__)


class BookDocumentsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)

        def documents_of_book(b):
            d_b = DatabaseBookDocuments.update_book_documents(b)
            d_b.to_file(b)
            return d_b.to_json()

        data = documents_of_book(book)
        return Response(data)

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    def put(self, request, book):
        book = DatabaseBook(book)
        ## Todo Mutex/lock
        obj = json.loads(request.body, encoding='utf-8')
        db = DatabaseBookDocuments.from_json(obj)
        db.to_file(book)
        logger.debug('Successfully saved DatabaseFile to {}'.format(book.local_path))

        return Response()


class BookDocumentsOdsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        filename = 'CM Default Metadatendatei'
        bytes = documents.database_documents.export_documents_to_xls(
            documents=documents.database_documents.documents,
            filename=filename,
            editor=str(request.user.username))
        return HttpResponse(bytes, content_type="application/vnd.oasis.opendocument.spreadsheet")


class DocumentView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        return Response(document.to_json())

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    def put(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        obj = json.loads(request.body, encoding='utf-8')
        doc_obj = Document.from_json(obj)
        documents.database_documents.documents[documents.database_documents.documents.index(document)] = doc_obj
        documents.to_file(book=book)
        return Response()


class DocumentsSVGView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document, width):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        pages = [DatabasePage(book, x) for x in document.pages_names]
        pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
        root = PcgtsToMonodiConverter(pcgts, document=document).root
        script_path = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'monodi_svg_render', 'bin', 'one-shot')
        proc = subprocess.run([script_path, "-", "-w", width], input=str(json.dumps(root.to_json())),
                              stdout=subprocess.PIPE, universal_newlines=True, stderr=subprocess.PIPE)
        str_result = proc.stdout
        reg = re.match(r".*(<svg.*</svg>).*", str_result, flags=re.DOTALL).group(1)
        return HttpResponse(reg, content_type="image/svg+xml")


class DocumentsMidiView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        pages = [DatabasePage(book, x) for x in document.pages_names]
        pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
        midi = SimpleMidiExporter(pcgts)
        seq = midi.generate_note_sequence(document=document)

        return Response(seq)


class DocumentOdsView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        filename = 'CM Default Metadatendatei'
        bytes = document.export_to_xls(filename, request.user.username)
        return HttpResponse(bytes, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


class MonodiConnectionView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def put(self, request, book):
        documents = json.loads(request.body.decode('utf-8'))
        documents = list(map(Document.from_json, documents))
        #pages = [DatabasePage(book, x) for x in document.pages_names]

        print(documents)
        if request.session.get('monodi_token', None) is not None:
            request.session['monodi_token'] = None

        else:
            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                            "Unauthorized. Login to the monodi service",
                            "Unauthorized. Login to the monodi service",
                            ErrorCodes.MONODI_LOGIN_REQUIRED,
                            ).response()
        return HttpResponse()


class MonodiLoginView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def post(self, request):
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        content = body['username']
        password = body['password']
        request.session['monodi_token'] = content
        return HttpResponse()

        r = python_request.post('https://editor.corpus-monodicum.de/api/login/login',
                                json={"user": content, "password": password})

        if r.status_code == 200:
            json_response = r.json()
            if json_response["kind"] == "LoginSuccessful":
                token = json_response["token"]
                request.session['monodi_token'] = token
                return HttpResponse()
        return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                        "No Account matches Credentials Found. Use other combination to login to the monodi service",
                        "No Account matches Credentials Found. Use other combination to login to the monodi service",
                        ErrorCodes.NO_MATCHING_CREDENTIALS_FOUND,
                        ).response()