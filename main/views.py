from django.http import HttpResponse, JsonResponse, HttpResponseNotModified, HttpResponseBadRequest
from .book import Book

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
