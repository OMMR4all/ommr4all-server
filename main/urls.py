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


# @login_required
def protected_serve(request, path, document_root=None, show_indexes=False):
    return serve(request, path, document_root, show_indexes)

def page_dir(book, page, root=settings.PRIVATE_MEDIA_ROOT):
    book = book.strip("/")
    page = page.strip('/')
    return os.path.join(root, book, page)

def list_pages(request, book: str):
    book = Book(book)
    pages = os.listdir(book.local_path())
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
    data = {
        'originalImageUrl': File(annotation_file.page, 'original').remote_path(),
        'binaryImageUrl': File(annotation_file.page, 'binary').remote_path(),
        'grayImageUrl': File(annotation_file.page, 'gray').remote_path(),
        'deskewedOriginalImageUrl': File(annotation_file.page, 'deskewed_original').remote_path(),
        'deskewedGrayImageUrl': File(annotation_file.page, 'deskewed_gray').remote_path(),
        'deskewedBinaryImageUrl': File(annotation_file.page, 'deskewed_binary').remote_path(),
        'detectedStaffsUrl': File(annotation_file.page, 'detected_staffs').remote_path(),
        'width': img.size[0],
        'height': img.size[1],
        'data': ''
    }
    if annotation_file.exists():
        with open(annotation_file.local_path(), 'r') as f:
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
    page = Page(Book(book), page)
    file = File(page, content)

    if not file.exists():
        file.create()

    return protected_serve(request, file.local_path(), "/", False)


urlpatterns = [
    re_path(r'^content/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    re_path(r'^storage/(?P<book>\w+)/(?P<page>\w+)/(?P<content>\w+)$', get_content),
    # re_path(r'^{0}(?P<path>.*)$'.format(settings.PRIVATE_MEDIA_URL.lstrip('/')),
    #   protected_serve, {'document_root': settings.PRIVATE_MEDIA_ROOT}),
    re_path(r'^listpages/(?P<book>\w+)$', list_pages),
    re_path(r'^annotation/(?P<book>\w+)/(?P<page>\w+)$', page_annotation),
    re_path(r'^save_page/(?P<book>\w+)/(?P<page>\w+)$', save_page),
    path('', views.index, name='index')
]
