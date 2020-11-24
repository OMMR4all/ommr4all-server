from django.urls import path, re_path

from restapi.views.bookaccess import BooksImportView, BookStatsView
from django.http import HttpResponse

from restapi.views.operation import OperationStatusView, OperationView, OperationTaskView
from restapi.views.bookaccess import BookView, BooksView, BookDownloaderView, BookUploadView, BookMetaView, \
    BookRenamePagesView
from restapi.views.pageaccess import PagePcGtsView, PageProgressView, PageStatisticsView, PageLockView
from restapi.views.virtualkeyboards import BookVirtualKeyboardView
from restapi.views.bookoperations import BookOperationStatusView, BookOperationTaskView, BookOperationView, BookPageSelectionView, BookOperationModelsView, BookOperationModelView
from restapi.views.auth import AuthView
from restapi.views.bookcomments import BookCommentsView, BookCommentsCountView
from restapi.views.bookpermissions import BookPermissionsView, BookUserPermissionsView, BookGroupPermissionsView, BookDefaultPermissionsView
from restapi.views.pageaccess import PageRenameView, PageProgressVerifyView, PageContentView
from restapi.views.user import UserBookPermissionsView
from restapi.views.bookstyles import BookStyleView, BookStylesView
from restapi.views.administrativedefaultmodels import AdministrativeDefaultModelsView
from restapi.views.tasks import TasksView, TaskView
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token


def ping(request):
    return HttpResponse()       # Just to check if server is up


urlpatterns = \
    [
        # jwt
        path('token-auth/', obtain_jwt_token, name='jwtAuth'),
        path('token-refresh/', refresh_jwt_token, name='jwtRefresh'),
        path('token-verify/', verify_jwt_token, name='jwtVerify'),

        # ping
        path('ping', ping),

        # auth
        re_path(r'^auth/(?P<auth>\w+)$', AuthView.as_view(), name='AuthView'),

       # user
        re_path(r'^user/book/(?P<book>\w+)/permissions$', UserBookPermissionsView.as_view()),

        # administrative
        re_path(r'^administrative/default_models/group/(?P<group>\w+)/style/(?P<style>.+)$', AdministrativeDefaultModelsView.as_view()),

        # styles
        re_path(r'^book-styles/(?P<id>.+)$', BookStyleView.as_view()),
        path('book-styles', BookStylesView.as_view()),

        # tasks
        path('tasks', TasksView.as_view()),
        re_path(r'^tasks/(?P<task_id>.+)$', TaskView.as_view()),

        # single book
        re_path(r'^book/(?P<book>\w+)/permissions/user/(?P<username>.+)$', BookUserPermissionsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/permissions/group/(?P<name>.+)$', BookGroupPermissionsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/permissions/default$', BookDefaultPermissionsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/permissions$', BookPermissionsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/comments/count', BookCommentsCountView.as_view()),
        re_path(r'^book/(?P<book>\w+)/comments', BookCommentsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/meta$', BookMetaView.as_view()),
        re_path(r'^book/(?P<book>\w+)/stats$', BookStatsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/upload/$', BookUploadView.as_view()),
        re_path(r'^book/(?P<book>\w+)/virtual_keyboard/$', BookVirtualKeyboardView.as_view()),
        re_path(r'^book/(?P<book>\w+)/rename_pages/$', BookRenamePagesView.as_view()),
        re_path(r'^book/(?P<book>\w+)/download/(?P<type>[\w\.]+)$', BookDownloaderView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/$', BookOperationView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/page_selection$', BookPageSelectionView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/task/(?P<task_id>[\w\-]+)$', BookOperationTaskView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/status$', BookOperationStatusView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/models$', BookOperationModelsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/operation/(?P<operation>\w+)/model/(?P<model>.+)$', BookOperationModelView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/content/pcgts$', PagePcGtsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/content/statistics$', PageStatisticsView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/content/page_progress/verify$', PageProgressVerifyView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/content/page_progress$', PageProgressView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/lock$', PageLockView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/content/(?P<content>\w+)$', PageContentView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/operation/(?P<operation>\w+)/$', OperationView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/operation/(?P<operation>\w+)/task/(?P<task_id>[\w\-]+)$', OperationTaskView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/rename$', PageRenameView.as_view()),
        re_path(r'^book/(?P<book>\w+)/page/(?P<page>\w+)/operation_status/(?P<operation>\w+)$', OperationStatusView.as_view()),
        re_path(r'^book/(?P<book>\w+)$', BookView.as_view()),

        # all books
        path('books/import', BooksImportView.as_view()),
        path('books', BooksView.as_view(), name='books'),
    ]
