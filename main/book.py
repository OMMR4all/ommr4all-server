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
    def __init__(self, id, output=[], requires=[]):
        self.id = id
        self.output = output
        self.requires = requires


file_definitions = {
    'original': FileDefinition(
        'original',
        ['original.jpg'],
    ),
    'binary': FileDefinition(
        'binary',
        ['binary.png'],
        requires=['gray'],
    ),
    'gray': FileDefinition(
        'gray',
        ['gray.png'],
        requires=['original'],
    ),
    'annotation': FileDefinition(
        'annotation',
        ['annotation.json'],
        requires=['original'],
    ),
    'preview': FileDefinition(
        'preview',
        ['preview.jpg'],
        requires=['original'],
    ),
    'deskewed_original': FileDefinition(
        'deskewed_original',
        ['deskewed_original.jpg', 'deskewed_gray.jpg', 'deskewed_binary.png'],
        requires=['binary', 'gray', 'original'],
    ),
    'deskewed_gray': FileDefinition(
        'deskewed_gray',
        ['deskewed_gray.jpg'],
        requires=['deskewed_original'],
    ),
    'deskewed_binary': FileDefinition(
        'deskewed_binary',
        ['deskewed_binary.png'],
        requires=['deskewed_original'],
    )

}

mutex_dict = LockedDict()

class File:
    def __init__(self, page: Page, fileId: str):
        self.page = page
        self.definition = file_definitions[fileId.strip('/')]

    def local_path(self, file_id=0):
        return os.path.join(self.page.local_path(), self.definition.output[file_id])

    def remote_path(self):
        return os.path.join(self.page.remote_path(), self.definition.id)

    def exists(self):
        return all(map(os.path.exists, [self.local_path(i) for i in range(len(self.definition.output))]))

    def create(self):
        with mutex_dict.get(self.local_path(), Lock()):
            if self.exists():
                # check if exists
                return

            # check if requirement files exist
            for file in self.definition.requires:
                File(self.page, file).create()

            # create local file
            logger.info('Creating local file {}'.format(self.local_path()))
            if self.definition.id == 'binary':
                from omr.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
                b = OCRopusBin()
                gray_image = File(self.page, 'gray').local_path()
                b.binarize(Image.open(gray_image)).save(self.local_path())
            elif self.definition.id == 'gray':
                from omr.preprocessing.gray.img2gray import im2gray
                im2gray(Image.open(File(self.page, 'original').local_path())).save(self.local_path())
            elif self.definition.id == 'preview':

                img = Image.open(File(self.page, 'original').local_path())
                img.thumbnail((200, 350))
                img.save(self.local_path())
            elif self.definition.id == 'deskewed_binary':
                logger.exception('Deskewed binary file does not exist, although deskewing was performed, using binary file')
                img = Image.open(File(self.page, 'binary').local_path())
                img.save(self.local_path())
            elif self.definition.id == 'deskewed_original':
                from omr.preprocessing.deskewer.deskewer import deskew
                orig, gray, binary = deskew(Image.open(File(self.page, 'original').local_path()),
                                            Image.open(File(self.page, 'gray').local_path()))
                orig.save(self.local_path(0))
                gray.save(self.local_path(1))
                binary.save(self.local_path(2))







