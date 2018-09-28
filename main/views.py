from django.http import HttpResponse, JsonResponse, HttpResponseNotModified, HttpResponseBadRequest
from .book import Book, Page, File
from omr.stafflines.text_line import TextLine
from omr.stafflines.json_util import json_to_line
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
import json


@csrf_exempt
def get_operation(request, book, page, operation):
    if operation == 'text_polygones':
        obj = json.loads(request.body, encoding='utf-8')
        initial_line = json_to_line(obj['points'])
        from omr.segmentation.text.extract_text_from_intersection import extract_text
        import pickle
        page = Page(Book(book), page)
        f = page.file('connected_components_deskewed')
        f.create()
        with open(f.local_path(), 'rb') as pkl:
            text_line = extract_text(pickle.load(pkl), initial_line)

        return JsonResponse(text_line.to_json())

    else:
        HttpResponseBadRequest()


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


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
