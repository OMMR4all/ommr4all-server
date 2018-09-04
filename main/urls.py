from django.urls import path, re_path
from django.contrib.auth.decorators import login_required
from django.views.static import serve
from django.conf import settings
from . import views
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
import os
import json
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt


# @login_required
def protected_serve(request, path, document_root=None, show_indexes=False):
    return serve(request, path, document_root, show_indexes)

def page_dir(book, page, root=settings.PRIVATE_MEDIA_ROOT):
    book = book.strip("/")
    page = page.strip('/')
    return os.path.join(root, book, page)

def list_pages(request, book: str):
    book = book.strip("/")
    book_dir = os.path.join(settings.PRIVATE_MEDIA_ROOT, book);
    pages = os.listdir(book_dir)
    data = {
        'pages': [
            {
                'id': page,
                'preview': os.path.join(page_dir(book, page, root=settings.PRIVATE_MEDIA_URL), 'original.jpg')
            } for page in pages]
    }
    return HttpResponse(json.dumps(data), content_type='application/json')


def page_annotation(request, book, page):
    annotation_file = os.path.join(page_dir(book, page), 'annotation.json')
    data = {
        'originalImageUrl': os.path.join(page_dir(book, page, root=settings.PRIVATE_MEDIA_URL), 'original.jpg'),
        'data': ''
    }
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            data['data'] = json.loads(f.read())

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
    page_d = page_dir(book, page)
    if content == 'original':
        return protected_serve(request, os.path.join(page_d, 'original.jpg'), "/", False)


urlpatterns = [
    re_path(r'^content/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    re_path(r'^{0}(?P<path>.*)$'.format(settings.PRIVATE_MEDIA_URL.lstrip('/')),
        protected_serve, {'document_root': settings.PRIVATE_MEDIA_ROOT}),
    re_path(r'^listpages/(?P<book>\w+)$', list_pages),
    re_path(r'^annotation/(?P<book>\w+)/(?P<page>\w+)$', page_annotation),
    re_path(r'^save_page/(?P<book>\w+)/(?P<page>\w+)$', save_page),
    path('', views.index, name='index')
]
