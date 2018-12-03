from json import JSONDecodeError

from django.http import HttpResponse, JsonResponse, HttpResponseNotModified, HttpResponseBadRequest, FileResponse

from omr.datatypes.performance.pageprogress import PageProgress
from .book import Book, Page, File, file_definitions, InvalidFileNameException
from omr.stafflines.text_line import TextLine
from omr.stafflines.json_util import json_to_line
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from omr.datatypes.pcgts import PcGts
from omr.datatypes.performance.statistics import Statistics
import json
from omr.datatypes.pcgts import PcGts
from PIL import Image
import numpy as np
import logging
import zipfile
import datetime
import os
import re

from main.operationworker import operation_worker, TaskDataStaffLineDetection, TaskStatusCodes


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
        task_data = TaskDataStaffLineDetection(page)
        if not operation_worker.put(task_data):
            status = operation_worker.status(task_data)
            if status.code == TaskStatusCodes.FINISHED:
                lines = operation_worker.pop_result(task_data)
                return JsonResponse({'status': status.to_json(), 'staffs': [l.to_json() for l in lines]})
            else:
                return JsonResponse({'status': status.to_json()})
        else:
            status = operation_worker.status(task_data)
            return JsonResponse({'status': status.to_json()})


    elif operation == 'symbols':
        from omr.symboldetection.predictor import PredictorParameters, PredictorTypes, create_predictor
        params = PredictorParameters(
            checkpoints=[page.book.local_path(os.path.join('pc_paths', 'model'))],
        )
        pred = create_predictor(PredictorTypes.PIXEL_CLASSIFIER, params)
        pcgts = PcGts.from_file(page.file('pcgts'))
        ps = list(pred.predict([pcgts]))
        music_lines = []
        for line_prediction in ps:
            music_lines.append({'symbols': [s.to_json() for s in line_prediction.symbols],
                                'id': line_prediction.line.operation.music_line.id})
        return JsonResponse({'musicLines': music_lines})

    elif operation == 'save_page_progress':
        obj = json.loads(request.body, encoding='utf-8')
        pp = PageProgress.from_json(obj)
        pp.to_json_file(page.file('page_progress').local_path())

        # add to backup archive
        with zipfile.ZipFile(page.file('page_progress_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('page_progress_{}.json'.format(datetime.datetime.now()), json.dumps(pp.to_json(), indent=2))

        return HttpResponse()
    elif operation == 'save_statistics':
        obj = json.loads(request.body, encoding='utf-8')
        total_stats = Statistics.from_json(obj)
        total_stats.to_json_file(page.file('statistics').local_path())

        # add to backup archive
        with zipfile.ZipFile(page.file('statistics_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('statistics_{}.json'.format(datetime.datetime.now()), json.dumps(total_stats.to_json(), indent=2))

        return HttpResponse()
    elif operation == 'save':
        obj = json.loads(request.body, encoding='utf-8')
        pcgts = PcGts.from_json(obj, page)
        pcgts.to_file(page.file('pcgts').local_path())

        # add to backup archive
        with zipfile.ZipFile(page.file('pcgts_backup').local_path(), 'a', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('pcgts_{}.json'.format(datetime.datetime.now()), json.dumps(pcgts.to_json(), indent=2))

        return HttpResponse()

    elif operation == 'clean':
        for key, _ in file_definitions.items():
            if key != 'color_original':
                File(page, key).delete()

        return HttpResponse()

    elif operation == 'train_symbol_detector':
        return JsonResponse({'response': 'started', 'bookState': {'symbolDetectionIsTraining': True}})

    else:
        return HttpResponseBadRequest()


def get_page_progress(request, book, page):
    page = Page(Book(book), page)
    file = File(page, 'page_progress')

    if not file.exists():
        file.create()

    try:
        return JsonResponse(PageProgress.from_json_file(file.local_path()).to_json())
    except JSONDecodeError as e:
        logging.error(e)
        file.delete()
        file.create()
        return JsonResponse(PageProgress.from_json_file(file.local_path()).to_json())


def get_pcgts(request, book, page):
    page = Page(Book(book), page)
    file = File(page, 'pcgts')

    if not file.exists():
        file.create()

    try:
        return JsonResponse(PcGts.from_file(file).to_json())
    except JSONDecodeError as e:
        logging.error(e)
        file.delete()
        file.create()
        return JsonResponse(PcGts.from_file(file).to_json())


def get_statistics(request, book, page):
    page = Page(Book(book), page)
    file = File(page, 'statistics')

    if not file.exists():
        file.create()

    try:
        return JsonResponse(Statistics.from_json_file(file.local_path()).to_json())
    except JSONDecodeError as e:
        logging.error(e)
        file.delete()
        file.create()
        return JsonResponse(Statistics.from_json_file(file.local_path()).to_json())


def list_book(request, book):
    book = Book(book)
    pages = book.pages()
    return JsonResponse({'pages': sorted([{'label': page.page} for page in pages if page.is_valid()], key=lambda v: v['label'])})


@csrf_exempt
def new_book(request):
    if request.method != 'POST':
        return HttpResponseBadRequest()

    book = json.loads(request.body, encoding='utf-8')
    if 'name' not in book:
        return HttpResponseBadRequest()

    book_id = re.sub('[^\w]', '_', book['name'])

    from .book_meta import BookMeta
    try:
        b = Book(book_id)
        if b.exists():
            return HttpResponseNotModified()

        if b.create(BookMeta(id=b.book, name=book['name'])):
            return JsonResponse(b.get_meta().to_json())
    except InvalidFileNameException as e:
        logging.error(e)
        return HttpResponse(status=InvalidFileNameException.STATUS)

    return HttpResponseBadRequest()


@csrf_exempt
def delete_book(request):
    if request.method != 'POST':
        return HttpResponseBadRequest()

    jdata = json.loads(request.body, encoding='utf-8')
    if 'id' not in jdata:
        return HttpResponseBadRequest()

    book_id = jdata['id']
    book = Book(book_id)
    book.delete()

    return HttpResponse()



def list_all_books(request):
    # TODO: sort by in request
    books = Book.list_available_book_metas()
    return JsonResponse({'books': sorted([book.to_json() for book in books], key=lambda b: b['name'])})


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
