from json import JSONDecodeError

from django.http import HttpResponse, JsonResponse, HttpResponseNotModified, HttpResponseBadRequest, FileResponse
from .book import Book, Page, File
from omr.stafflines.text_line import TextLine
from omr.stafflines.json_util import json_to_line
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from omr.datatypes.pcgts import PcGts
import json
from omr.datatypes.pcgts import PcGts
from PIL import Image
import numpy as np
import logging


@csrf_exempt
def get_operation(request, book, page, operation):
    page = Page(Book(book), page)
    if operation == 'text_polygones':
        obj = json.loads(request.body, encoding='utf-8')
        initial_line = json_to_line(obj['points'])
        from omr.segmentation.text.extract_text_from_intersection import extract_text
        import pickle
        f = page.file('connected_components_deskewed')
        f.create()
        with open(f.local_path(), 'rb') as pkl:
            text_line = extract_text(pickle.load(pkl), initial_line)

        return JsonResponse(text_line.to_json())

    elif operation == 'staffs':
        from omr.stafflines.detection.dummy_detector import detect
        binary = Image.open(File(page, 'binary_deskewed').local_path())
        gray = Image.open(File(page, 'gray_deskewed').local_path())
        lines = detect(np.array(binary) // 255, np.array(gray) / 255)
        return JsonResponse({'staffs': [l.to_json() for l in lines]})

    elif operation == 'save':
        obj = json.loads(request.body, encoding='utf-8')
        pcgts = PcGts.from_json(obj)
        pcgts.to_file(page.file('pcgts').local_path())

        return HttpResponse()

    else:
        HttpResponseBadRequest()


def get_pcgts(request, book, page):
    page = Page(Book(book), page)
    file = File(page, 'pcgts')

    if not file.exists():
        file.create()

    try:
        return JsonResponse(PcGts.from_file(file.local_request_path()).to_json())
    except JSONDecodeError as e:
        logging.error(e)
        file.delete()
        file.create()
        return JsonResponse(PcGts.from_file(file.local_request_path()).to_json())


def list_book(request, book):
    book = Book(book)
    pages = book.pages()
    return JsonResponse({'pages': sorted([{'label': page.page} for page in pages if page.is_valid()], key=lambda v: v['label'])})


def new_book(request, book):
    book = Book(book)
    if book.exists():
        return HttpResponseNotModified()

    if book.create():
        return JsonResponse({'label': book.book})

    return HttpResponseBadRequest()


def list_all_books(request):
    books = Book.list_available()
    return JsonResponse({'books': sorted([{'label': book.book} for book in books if book.is_valid()], key=lambda v: v['label'])})


def book_download(request, book, type):
    book = Book(book)
    if type == 'annotations.zip':
        import zipfile, io, os
        s = io.BytesIO()
        zf = zipfile.ZipFile(s, 'w')
        pages = book.pages()
        for page in pages:
            color_img = page.file('color_deskewed')
            binary_img = page.file('binary_deskewed')
            annotation = page.file('annotation')
            if not color_img.exists() or not binary_img.exists() or not annotation.exists():
                continue

            zf.write(color_img.local_path(), os.path.join('color', page.page + color_img.ext()))
            zf.write(binary_img.local_path(), os.path.join('binary', page.page + binary_img.ext()))
            zf.write(annotation.local_path(), os.path.join('annotation', page.page + annotation.ext()))

        zf.close()
        s.seek(0)
        return FileResponse(s, as_attachment=True, filename=book.book + '.zip')

    return HttpResponseBadRequest()



def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
