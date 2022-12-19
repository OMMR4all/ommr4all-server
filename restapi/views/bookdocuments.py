import base64
import io
import json
import os
import re
import subprocess

from PIL import Image
from django.views.decorators.csrf import csrf_exempt

from database.database_dictionary import DatabaseDictionary
from database.file_formats.book import document
from database.file_formats.book.documents import Documents
from database.file_formats.exporter.midi.simple_midi import SimpleMidiExporter

from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from database import *
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats import PcGts
import logging
import json

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


class BookDictionaryView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book):
        book = DatabaseBook(book)
        b = DatabaseDictionary.load(book)
        return Response(b.to_json())

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    def put(self, request, book):
        book = DatabaseBook(book)
        ## Todo Mutex/lock
        obj = json.loads(request.body, encoding='utf-8')
        db = DatabaseDictionary.from_json(obj)
        db.to_file(book)
        logger.debug('Successfully saved DatabaseFile to {}'.format(book.local_path))

        return Response()


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

class BookPageDocumentsUpdateView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, page):
        book = DatabaseBook(book)
        page= DatabasePage(page=page, book=book)

        documents = DatabaseBookDocuments().load(book)

        def documents_of_book(documents, book, page):

            d_b = documents.update_documents_of_page(page=page, book=book)

            d_b.to_file(book)

            return d_b.to_json()

        data = documents_of_book(documents, book, page)

        return Response(data)



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

class DocumentImageLineImageView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document, line):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        #response = HttpResponse(content_type="image/png")
        img = Image.fromarray(document.get_image_of_document_by_line(book, line))
        #img.save(response, "PNG")
        buf = io.BytesIO()
        img.save(buf, "png")
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return HttpResponse(img_b64)

class DocumentImageLineTextView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document, line):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        text = document.get_text_of_document_by_line(book, line)

        return Response(text)

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

class DocumentPCGTSUpdatesView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.SAVE])
    def put(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        ## Todo Mutex/lock
        obj = json.loads(request.body, encoding='utf-8')
        document.update_pcgts(book=book, lines=obj)
        #db = DatabaseBookDocuments.from_json(obj)
        #db.to_file(book)
        logger.debug('Successfully updated document text of DatabaseFile to {}'.format(book.local_path))

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


class DocumentsSimilarityView(APIView):
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
        editor = request.session.get('monodi_user', None)
        user = editor if editor is not None else request.user.username
        bytes = document.export_to_xls(filename, user)
        return HttpResponse(bytes, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

class DocumentImageView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def get(self, request, book, document):
        book = DatabaseBook(book)
        documents = DatabaseBookDocuments().load(book)
        document: Document = documents.database_documents.get_document_by_id(document)
        filename = 'CM Default Metadatendatei'
        editor = request.session.get('monodi_user', None)
        user = editor if editor is not None else request.user.username
        bytes = document.export_to_xls(filename, user)
        return HttpResponse(bytes, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

class MonodiConnectionView(APIView):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @require_permissions([DatabaseBookPermissionFlag.READ])
    def put(self, request, book):
        book = DatabaseBook(book)
        documents = json.loads(request.body.decode('utf-8'))
        documents = list(map(Document.from_json, documents))
        # request.session['monodi_token'] = None

        if request.session.get('monodi_token', None) is not None:
            editor = request.session.get('monodi_user', None)
            import base64
            filename = 'CM Default Metadatendatei'

            bytes = Documents.export_documents_to_xls(
                documents=documents,
                filename=filename,
                editor=str(editor))
            base64EncodedStr = base64.b64encode(bytes).decode()

            # Convert it to valid json
            base64EncodedStr = "".join(["\"", base64EncodedStr, "\""])
            header = {'Content-Type': 'text/plain', 'Authorization': '{}'.format(request.session.get('monodi_token',
                                                                                                     None))}
            import_documents_response = python_request.post(
                url='https://editor.corpus-monodicum.de/api/source/importDocuments',
                data=base64EncodedStr,
                headers=header)

            if import_documents_response.status_code == 200:
                json_response = import_documents_response.json()
                if json_response["kind"] == "UploadFinished":
                    header = {'Authorization': '{}'.format(request.session.get('monodi_token', None))}
                    for document in documents:
                        pages = [DatabasePage(book, x) for x in document.pages_names]
                        pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in
                                 pages]
                        root = PcgtsToMonodiConverter(pcgts, document=document)
                        json_data = root.get_Monodi_json(document=document, editor=str(editor))

                        def pp_json(json_thing, sort=True, indents=4):
                            if type(json_thing) is str:
                                print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
                            else:
                                print(json.dumps(json_thing, sort_keys=sort, indent=indents))
                            return None

                        doc_response = python_request.post('https://editor.corpus-monodicum.de/api/document/update',
                                                           json=json_data, headers=header)
                        # with open("/tmp/demo.json", "w") as fp:
                        #    #js = json.loads(json_data)
                        #    json.dump(json_data, fp, sort_keys=False, indent=4)
                        if doc_response.status_code == 200:
                            json_response = doc_response.json()

                            if json_response["kind"] == "Ok":
                                continue
                            else:
                                return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                                                "Error when importing documents to Monodi",
                                                "Error when importing documents to Monodi",
                                                ErrorCodes.ERROR_ON_UPDATING_DOCUMENT,
                                                ).response()
                        else:
                            return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                                            "Error when importing documents to Monodi",
                                            "Error when importing documents to Monodi",
                                            ErrorCodes.ERROR_ON_UPDATING_DOCUMENT,
                                            ).response()
                    return HttpResponse()

                else:
                    return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                                    "Error when importing documents to Monodi",
                                    "Error when importing documents to Monodi",
                                    ErrorCodes.ERROR_ON_IMPORTING_DOCUMENTS,
                                    ).response()

            else:
                return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                                "Error when importing documents to Monodi",
                                "Error when importing documents to Monodi",
                                ErrorCodes.ERROR_ON_IMPORTING_DOCUMENTS,
                                ).response()
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
        # request.session['monodi_token'] = content
        # return HttpResponse()

        r = python_request.post('https://editor.corpus-monodicum.de/api/login/login',
                                json={"user": content, "password": password})

        if r.status_code == 200:
            json_response = r.json()
            if json_response["kind"] == "LoginSuccessful":
                token = json_response["token"]
                request.session['monodi_user'] = content
                request.session['monodi_token'] = token
                return HttpResponse()
        return APIError(status.HTTP_406_NOT_ACCEPTABLE,
                        "No Account matches credentials. Use other combination to login to the monodi service",
                        "No Account matches credentials. Use other combination to login to the monodi service",
                        ErrorCodes.NO_MATCHING_CREDENTIALS_FOUND,
                        ).response()
