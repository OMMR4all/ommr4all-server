from django.urls import path, re_path
from django.contrib.auth.decorators import login_required
from django.views.static import serve
from django.conf import settings
from . import views
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
import os
import json
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from .book import *
from PIL import Image


# @login_required
def protected_serve(request, path, document_root=None, show_indexes=False):
    return serve(request, path, document_root, show_indexes)

def page_dir(book, page, root=settings.PRIVATE_MEDIA_ROOT):
    book = book.strip("/")
    page = page.strip('/')
    return os.path.join(root, book, page)

def list_pages(request, book: str):
    book = Book(book)
    if os.path.exists(book.local_path()) and os.path.isdir(book.local_path()):
        pages = os.listdir(book.local_path())
    else:
        pages = []

    data = {
        'pages': [
            {
                'id': page,
                'preview': File(Page(book, page), 'preview').remote_path(),
            } for page in pages]
    }
    return HttpResponse(json.dumps(data), content_type='application/json')


def page_annotation(request, book, page):
    annotation_file = File(Page(Book(book), page), 'annotation')
    img = Image.open(File(annotation_file.page, 'deskewed_original').local_path())
    files = {}
    for label, definition in file_definitions.items():
        files[label] = File(annotation_file.page, label).remote_path()

    data = {
        'files': files,
        'width': img.size[0],
        'height': img.size[1],
        'annotation_data': {},
    }
    if annotation_file.exists():
        with open(annotation_file.local_path(), 'r') as f:
            data['annotation_data'] = json.load(f)

    return JsonResponse(data)


@csrf_exempt
def save_page(request, book, page):
    if request.method == 'POST':
        annotation_file = os.path.join(page_dir(book, page), 'annotation.json')
        with open(annotation_file, 'w') as f:
            f.write(request.body.decode('utf-8'))

        return JsonResponse({'status': 'ok'})

    return HttpResponseBadRequest()


def get_content(request, book, page, content):
    page = Page(Book(book), page)
    file = File(page, content)

    if not file.exists():
        file.create()

    return protected_serve(request, file.local_request_path(), "/", False)

@csrf_exempt
def upload_to_book(request, book):
    if request.method == 'POST':
        book = Book(book)
        if not os.path.exists(book.local_path()):
            os.mkdir(book.local_path())

        for type, file in request.FILES.items():
            name = os.path.splitext(os.path.basename(file.name))[0].replace(" ", "_").replace("-", "_").replace('.', '_')
            page = Page(book, name)
            if not os.path.exists(page.local_path()):
                os.mkdir(page.local_path())

            type = file.content_type
            if not type.startswith('image/'):
                return HttpResponseBadRequest()

            try:
                img = Image.open(file.file, 'r').convert('RGB')
                original = File(page, 'color_original')
                img.save(original.local_path())

            except:
                return HttpResponseBadRequest()


        return HttpResponse()

    return HttpResponseBadRequest()

urlpatterns = [
    re_path(r'^content/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    re_path(r'^storage/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    # re_path(r'^{0}(?P<path>.*)$'.format(settings.PRIVATE_MEDIA_URL.lstrip('/')),
    #   protected_serve, {'document_root': settings.PRIVATE_MEDIA_ROOT}),
    re_path(r'^listpages/(?P<book>\w+)$', list_pages),
    re_path(r'^annotation/(?P<book>\w+)/(?P<page>\w+)$', page_annotation),
    re_path(r'^save_page/(?P<book>\w+)/(?P<page>\w+)$', save_page),
    re_path(r'^book/(?P<book>\w+)/upload/$', upload_to_book),
    re_path(r'^book/(?P<book>\w+)/list/$', views.list_book),
    re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/save$', save_page),
    re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/content/(?P<content>\w+)$', get_content),
    re_path(r'^book/(?P<book>\w+)/(?P<page>\w+)/operation/(?P<operation>\w+)$', views.get_operation),
    path('books/list', views.list_all_books, name='list_all_books'),
    re_path(r'^books/new/(?P<book>\w+)$', views.new_book, name='new_book'),
    path('', views.index, name='index'),
]
