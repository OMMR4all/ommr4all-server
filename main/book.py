from django.conf import settings
import os
from PIL import Image
from multiprocessing import Lock
from locked_dict.locked_dict import LockedDict
import logging
logger = logging.getLogger(__name__)


class Book:
    def __init__(self, book: str):
        self.book = book.strip('/')

    def page(self, page):
        return Page(self, page)

    def local_path(self):
        return os.path.join(settings.PRIVATE_MEDIA_ROOT, self.book)

    def remote_path(self):
        return os.path.join(settings.PRIVATE_MEDIA_URL, self.book)


class Page:
    def __init__(self, book: Book, page: str):
        self.book = book
        self.page = page.strip("/")

    def file(self, fileId):
        return File(self, fileId)

    def local_path(self):
        return os.path.join(self.book.local_path(), self.page)

    def remote_path(self):
        return os.path.join(self.book.remote_path(), self.page)


class FileDefinition:
    def __init__(self, id, filename):
        self.id = id
        self.filename = filename


file_definitions = {
    'original': FileDefinition(
        'original',
        'original.jpg',
    ),
    'binary': FileDefinition(
        'binary',
        'binary.png',
    ),
    'annotation': FileDefinition(
        'annotation',
        'annotation.json',
    ),
    'preview': FileDefinition(
        'preview',
        'preview.jpg',
    ),
}

mutex_dict = LockedDict()

class File:
    def __init__(self, page: Page, fileId: str):
        self.page = page
        self.definition = file_definitions[fileId.strip('/')]

    def local_path(self):
        return os.path.join(self.page.local_path(), self.definition.filename)

    def remote_path(self):
        return os.path.join(self.page.remote_path(), self.definition.id)

    def exists(self):
        return os.path.exists(self.local_path())

    def create(self):
        with mutex_dict.get(self.local_path(), Lock()):
            if self.exists():
                # check if exists
                return

            logger.info('Creating local file {}'.format(self.local_path()))
            if self.definition.id == 'binary':
                from omr.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
                b = OCRopusBin()
                original_image = File(self.page, 'original').local_path()
                b.binarize(Image.open(original_image)).save(self.local_path())
            elif self.definition.id == 'preview':

                img = Image.open(File(self.page, 'original').local_path())
                img.thumbnail((200, 350))
                img.save(self.local_path())




