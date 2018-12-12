from django.urls import path, re_path
from django.views.static import serve
from . import views
from django.http import HttpResponse
from .book import *
from main.api import OperationStatusView, OperationView, BookView, BooksView, \
    PageProgressView, PageStatisticsView, PagePcGtsView, BookDownloaderView, BookUploadView


# @login_required
def protected_serve(request, path, document_root=None, show_indexes=False):
    return serve(request, path, document_root, show_indexes)


def get_content(request, book, page, content):
    page = Page(Book(book), page)
    file = File(page, content)

    if not file.exists():
        file.create()

    return protected_serve(request, file.local_request_path(), "/", False)


def ping(request):
    return HttpResponse()       # Just to check if server is up


urlpatterns = [
    # ping
    path('api/ping', ping),

    # content
    re_path(r'^api/content/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    re_path(r'^api/storage/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),

    # single book
    re_path(r'^api/book/(?P<book>\w+)/upload/$', BookUploadView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/download/(?P<type>[\w\.]+)$', BookDownloaderView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/content/pcgts$', PagePcGtsView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/content/statistics$', PageStatisticsView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/content/page_progress$', PageProgressView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/content/(?P<content>\w+)$', get_content),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/operation/(?P<operation>\w+)$', OperationView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)/(?P<page>\w+)/operation_status/(?P<operation>\w+)$', OperationStatusView.as_view()),
    re_path(r'^api/book/(?P<book>\w+)$', BookView.as_view()),

    # all books
    path('api/books', BooksView.as_view(), name='books'),

    # static pages/webapp
    path('', views.index),
    re_path(r'^(?P<path>.*)/$', views.index, name='index'),
]
