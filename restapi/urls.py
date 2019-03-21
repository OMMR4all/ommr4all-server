from django.urls import path, re_path
from django.views.static import serve
from . import views
from django.http import HttpResponse
from database import *
from restapi.api import OperationStatusView, OperationView, BookView, BooksView, \
    PageProgressView, PageStatisticsView, PagePcGtsView, BookDownloaderView, BookUploadView, BookMetaView, \
    BookVirtualKeyboardView
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token


@api_view(['GET'])
@permission_classes((IsAuthenticated, ))
def protected_serve(request, path, document_root=None, show_indexes=False):
    return serve(request._request, path, document_root, show_indexes)


@api_view(['GET'])
@permission_classes((IsAuthenticated, ))
def get_content(request, book, page, content):
    page = DatabasePage(DatabaseBook(book), page)
    file = DatabaseFile(page, content)

    if not file.exists():
        file.create()

    return protected_serve(request._request, file.local_request_path(), "/", False)


def ping(request):
    return HttpResponse()       # Just to check if server is up


urlpatterns = \
    [
        # jwt
        path('token-auth/', obtain_jwt_token),
        path('token-refresh/', refresh_jwt_token),
        path('token-verify/', verify_jwt_token),

        # ping
        path('ping', ping),

        # content
        re_path(r'^content/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
        re_path(r'^storage/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),

        # single book
        re_path(r'^book/(?P<book>\w+)/meta$', BookMetaView.as_view()),
        re_path(r'^book/(?P<book>\w+)/upload/$', BookUploadView.as_view()),
        re_path(r'^book/(?P<book>\w+)/virtual_keyboard/$', BookVirtualKeyboardView.as_view()),
        re_path(r'^book/(?P<book>\w+)/download/(?P<type>[\w\.]+)$', BookDownloaderView.as_view()),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/content/pcgts$', PagePcGtsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/content/statistics$', PageStatisticsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/content/page_progress$', PageProgressView.as_view()),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/content/(?P<content>\w+)$', get_content),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/operation/(?P<operation>\w+)$', OperationView.as_view()),
        re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/operation_status/(?P<operation>\w+)$', OperationStatusView.as_view()),
        re_path(r'^book/(?P<book>\w+)$', BookView.as_view()),

        # all books
        path('books', BooksView.as_view(), name='books'),
    ] \
