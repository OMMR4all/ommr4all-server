from django.conf import settings
import os
from PIL import Image
from multiprocessing import Lock
from locked_dict.locked_dict import LockedDict
import numpy as np
import logging
from typing import NamedTuple, List, Tuple
from ommr4all.settings import PRIVATE_MEDIA_ROOT, PRIVATE_MEDIA_URL


logger = logging.getLogger(__name__)
import re


file_name_validator = re.compile('\w+')


class Book:
    @staticmethod
    def list_available():
        return [Book(name) for name in os.listdir(PRIVATE_MEDIA_ROOT)]

    def __init__(self, book: str):
        self.book = book.strip('/')

    def pages(self):
        assert(self.is_valid())

        pages = [Page(self, p) for p in sorted(os.listdir(self.local_path('pages')))]
        return [p for p in pages if p.is_valid()]

    def page(self, page):
        return Page(self, page)

    def local_path(self, sub=''):
        return os.path.join(PRIVATE_MEDIA_ROOT, self.book, sub)

    def remote_path(self):
        return os.path.join(PRIVATE_MEDIA_URL, self.book)

    def is_valid_name(self):
        return file_name_validator.fullmatch(self.book)

    def is_valid(self):
        if not self.is_valid_name():
            return False

        if not os.path.exists(self.local_path()):
            return True

        if not os.path.isdir(self.local_path()):
            return False

        return True

    def exists(self):
        return os.path.exists(self.local_path()) and os.path.isdir(self.local_path())

    def create(self):
        if self.exists():
            return True

        if not self.is_valid():
            return False

        os.mkdir(self.local_path())
        return True


class Page:
    def __init__(self, book: Book, page: str):
        self.book = book
        self.page = page.strip("/")

    def file(self, fileId, create_if_not_existing=False):
        return File(self, fileId, create_if_not_existing)

    def local_file_path(self, f):
        return os.path.join(self.local_path(), f)

    def local_path(self):
        return os.path.join(self.book.local_path('pages'), self.page)

    def remote_path(self):
        return os.path.join(self.book.remote_path(), self.page)

    def is_valid(self):
        if not os.path.exists(self.local_path()):
            return True

        if not os.path.isdir(self.local_path()):
            return False

        return True


class PageMeta:
    def __init__(self, d: dict = None):
        d = d if d else {}
        self.width = d.get('width', -1)
        self.height = d.get('height', -1)

    def json(self):
        return {
            'width': self.width,
            'height': self.height,
        }

    def dumpfn(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.json(), f)

    @staticmethod
    def loadfn(filename):
        import json
        with open(filename, 'r') as f:
            return PageMeta(json.load(f))


class FileDefinition(NamedTuple):
    id: str
    output: List[str] = []
    requires: List[str] = []
    default: int = 0
    has_preview: bool = False


file_definitions = {
    'statistics': FileDefinition(
        'statistics',
        ['statistics.json'],
    ),
    'statistics_backup': FileDefinition(
        'statistics_backup',
        ['statistics_backup.zip'],
    ),
    'page_progress': FileDefinition(
        'page_progress',
        ['page_progress.json'],
    ),
    'page_progress_backup': FileDefinition(
        'page_progress_backup',
        ['page_progress_backup.zip'],
    ),
    'meta': FileDefinition(
        'meta',
        ['meta.json'],
        requires=['color_original'],
    ),
    'color_original': FileDefinition(
        'color_original',
        ['color_original.jpg'],
        has_preview=True,
    ),
    'binary_original': FileDefinition(
        'binary_original',
        ['binary_original.png'],
        requires=['gray_original'],
        has_preview=True,
    ),
    'gray_original': FileDefinition(
        'gray_original',
        ['gray_original.png'],
        requires=['color_original'],
        has_preview=True,
    ),
    'annotation': FileDefinition(
        'annotation',
        ['annotation.json'],
        requires=['color_original'],
    ),
    'pcgts': FileDefinition(
        'pcgts',
        ['pcgts.json'],
        requires=['color_original'],
    ),
    'pcgts_backup': FileDefinition(
        'pcgts_backup',
        ['pcgts_backup.zip'],
    ),
    'color_deskewed': FileDefinition(
        'color_deskewed',
        ['color_deskewed.jpg', 'gray_deskewed.jpg', 'binary_deskewed.png'],
        requires=['binary_original', 'gray_original', 'color_original'],
        has_preview=True,
    ),
    'gray_deskewed': FileDefinition(
        'gray_deskewed',
        ['gray_deskewed.jpg'],
        requires=['color_deskewed'],
        has_preview=True,
    ),
    'binary_deskewed': FileDefinition(
        'binary_deskewed',
        ['binary_deskewed.png'],
        requires=['color_deskewed'],
        has_preview=True,
    ),
    'connected_components_deskewed': FileDefinition(
        'connected_components_deskewed',
        ['connected_components_deskewed.pkl'],
        requires=['binary_deskewed'],
    ),
    'color_cropped': FileDefinition(
        'color_cropped',
        ['color_cropped.jpg', 'gray_cropped.jpg', 'binary_cropped.png'],
        requires=['color_deskewed', 'gray_deskewed', 'binary_deskewed', 'annotation'],
        has_preview=True,

    ),
    'gray_cropped': FileDefinition(
        'gray_cropped',
        ['color_cropped.jpg', 'gray_cropped.jpg', 'binary_cropped.png'],
        default=1,
        requires=['color_cropped'],
        has_preview=True,
    ),
    'binary_cropped': FileDefinition(
        'binary_cropped',
        ['color_cropped.jpg', 'gray_cropped.jpg', 'binary_cropped.png'],
        default=2,
        requires=['color_cropped'],
        has_preview=True,
    ),
    'detected_staffs': FileDefinition(
        'detected_staffs',
        ['detected_staffs.json'],
        requires=['binary_deskewed', 'gray_deskewed'],
    ),
    'color_detected_staffs': FileDefinition(
        'color_detected_staffs',
        ['color_detected_staffs.jpg'],
        requires=['detected_staffs', 'color_cropped'],
        has_preview=True,
    ),
    'gray_detected_staffs': FileDefinition(
        'gray_detected_staffs',
        ['gray_detected_staffs.jpg'],
        requires=['detected_staffs', 'gray_cropped'],
        has_preview=True,
    ),
    'binary_detected_staffs': FileDefinition(
        'binary_detected_staffs',
        ['binary_detected_staffs.jpg'],
        requires=['detected_staffs', 'binary_cropped'],
        has_preview=True,
    ),
    'dewarped_original': FileDefinition(
        'dewarped_original',
        ['dewarped_original.jpg', 'dewarped_gray.jpg', 'dewarped_binary.png'],
        requires=['cropped_binary', 'cropped_gray', 'cropped_original', 'annotation'],
        has_preview=True,
    ),
    'dewarped_gray': FileDefinition(
        'dewarped_gray',
        ['dewarped_gray.jpg'],
        requires=['dewarped_original'],
        has_preview=True,
    ),
    'dewarped_binary': FileDefinition(
        'deskewed_binary',
        ['dewarped_binary.png'],
        requires=['dewarped_original'],
        has_preview=True,
    ),

}

mutex_dict = LockedDict()

thumbnail_size = (200, 350)


class File:
    def __init__(self, page: Page, fileId: str, create_if_not_existing=False):
        self.page = page
        self._fileId = fileId.strip('/')
        if self._fileId.endswith('_preview'):
            self.preview = True
            self.definition: FileDefinition = file_definitions[self._fileId[:-len('_preview')]]
        else:
            self.preview = False
            self.definition: FileDefinition = file_definitions[fileId.strip('/')]

        if create_if_not_existing and not self.exists():
            self.create()

    def local_path(self, file_id=-1):
        return os.path.join(self.page.local_path(), self.definition.output[file_id if file_id >= 0 else self.definition.default])

    def local_thumbnail_path(self, file_id=-1):
        return os.path.splitext(self.local_path(file_id))[0] + '_preview.jpg'

    def local_request_path(self):
        if self.preview:
            return self.local_thumbnail_path()
        else:
            return self.local_path()

    def ext(self):
        return os.path.splitext(self.local_request_path())[-1]

    def remote_path(self):
        if self.preview:
            return os.path.join(self.page.remote_path(), self.definition.id + '_preview')
        else:
            return os.path.join(self.page.remote_path(), self.definition.id)

    def exists(self):
        return all(map(os.path.exists, [self.local_path(i) for i in range(len(self.definition.output))])) \
               and (not self.definition.has_preview or all(map(os.path.exists, [self.local_thumbnail_path(i) for i in range(len(self.definition.output))])))

    def delete(self):
        for i in range(len(self.definition.output)):
            if os.path.exists(self.local_path(file_id=i)):
                os.remove(self.local_path(file_id=i))
            if os.path.exists(self.local_thumbnail_path(file_id=i)):
                os.remove(self.local_thumbnail_path(file_id=i))

    def create(self):
        from omr.datatypes import MusicLines

        with mutex_dict.get(self.local_path(), Lock()):
            if self.exists():
                # check if exists
                return

            # check if requirement files exist
            for file in self.definition.requires:
                File(self.page, file).create()

            # check again if exists since the requirements might have created that file!
            if self.exists():
                return

            # create local file
            logger.info('Creating local file {}'.format(self.local_path()))
            if self.definition.id == 'statistics' \
                    or self.definition.id == 'page_progress':
                import json
                with open(self.local_path(), 'w') as f:
                    json.dump({}, f)
            elif self.definition.id == 'page_progress_backup' \
                    or self.definition.id == 'statistics_backup':
                import zipfile
                zf = zipfile.ZipFile(self.local_path(), mode='w', compression=zipfile.ZIP_DEFLATED)
                zf.close()
            elif self.definition.id == 'annotation':
                import json
                with open(self.local_path(), 'w') as f:
                    json.dump({}, f)
            elif self.definition.id == 'pcgts':
                from omr.datatypes.pcgts import PcGts, Page, Meta
                img = Image.open(File(self.page, 'color_original').local_path())
                pcgts = PcGts(
                    meta=Meta(),
                    page=Page(location=self.page),
                )
                pcgts.page.image_width, pcgts.page.image_height = img.size
                pcgts.to_file(self.local_path())
            elif self.definition.id == 'pcgts_backup':
                import zipfile
                zf = zipfile.ZipFile(self.local_path(), mode='w', compression=zipfile.ZIP_DEFLATED)
                zf.close()
            elif self.definition.id == 'meta':
                img = Image.open(File(self.page, 'color_original').local_path())
                width, height = img.size
                meta = PageMeta({
                    'width': width,
                    'height': height,
                })
                meta.dumpfn(self.local_path())
            elif self.definition.id == 'color_original':
                # create preview
                img = Image.open(self.local_path())
                img.thumbnail(thumbnail_size)
                img.save(self.local_thumbnail_path())
            elif self.definition.id == 'binary_original':
                from omr.preprocessing.binarizer.ocropus_binarizer import OCRopusBin
                b = OCRopusBin()
                gray_image = File(self.page, 'gray_original').local_path()
                binary = b.binarize(Image.open(gray_image))
                binary.save(self.local_path())
                binary.thumbnail(thumbnail_size)
                binary.save(self.local_thumbnail_path())
            elif self.definition.id == 'gray_original':
                from omr.preprocessing.gray.img2gray import im2gray
                gray = im2gray(Image.open(File(self.page, 'color_original').local_path()))
                gray.save(self.local_path())
                gray.thumbnail(thumbnail_size)
                gray.save(self.local_thumbnail_path())
            elif self.definition.id == 'color_deskewed':
                from omr.preprocessing.deskewer import default_deskewer
                deskewer = default_deskewer()
                orig, gray, binary = deskewer.deskew(
                    Image.open(File(self.page, 'color_original').local_path()),
                    Image.open(File(self.page, 'gray_original').local_path()),
                    Image.open(File(self.page, 'binary_original').local_path()),
                )
                orig.save(self.local_path(0))
                gray.save(self.local_path(1))
                binary.save(self.local_path(2))
                orig.thumbnail(thumbnail_size)
                orig.save(self.local_thumbnail_path(0))
                gray.thumbnail(thumbnail_size)
                gray.save(self.local_thumbnail_path(1))
                binary.thumbnail(thumbnail_size)
                binary.save(self.local_thumbnail_path(2))
            elif self.definition.id == 'connected_components_deskewed':
                import pickle
                from omr.preprocessing.util.connected_compontents import connected_compontents_with_stats
                binary = np.array(Image.open(File(self.page, 'binary_deskewed').local_path()))
                with open(self.local_path(), 'wb') as f:
                    pickle.dump(connected_compontents_with_stats(binary), f)
            elif self.definition.id == 'detected_staffs':
                from omr.stafflines.detection.dummy_detector import detect
                import json
                binary = Image.open(File(self.page, 'binary_deskewed').local_path())
                gray = Image.open(File(self.page, 'gray_deskewed').local_path())
                lines = detect(np.array(binary) // 255, np.array(gray) / 255)
                s = lines.to_json()
                with open(self.local_path(), 'w') as f:
                    json.dump(s, f, indent=2)
            elif self.definition.id == 'color_cropped':
                from omr.preprocessing.cropper import default_cropper
                import json
                images = (
                    Image.open(File(self.page, 'color_deskewed').local_path()),
                    Image.open(File(self.page, 'gray_deskewed').local_path()),
                    Image.open(File(self.page, 'binary_deskewed').local_path()),
                )
                cropper = default_cropper()
                (orig, gray, binary) = cropper.crop(images[0], images[1], images[2])
                orig.save(self.local_path(0))
                orig.thumbnail(thumbnail_size)
                orig.save(self.local_thumbnail_path(0))
                gray.save(self.local_path(1))
                gray.thumbnail(thumbnail_size)
                gray.save(self.local_thumbnail_path(1))
                binary.save(self.local_path(2))
                binary.thumbnail(thumbnail_size)
                binary.save(self.local_thumbnail_path(2))

                # annotation = File(self.page, 'annotation').local_path()
                # s = MusicLines.from_json(json.load(open(annotation, 'r')))
                # s.crop = cropper.rect
                # with open(annotation, 'w') as f:
                #    json.dump(s.to_json(), f, indent=2)
            elif self.definition.id == 'color_detected_staffs':
                import json
                img = Image.open(File(self.page, 'color_deskewed').local_path())
                with open(File(self.page, 'detected_staffs').local_path(), 'r') as f:
                    staffs = MusicLines.from_json(json.load(f))
                    img = np.array(img)
                    staffs.draw(img)
                    img = Image.fromarray(img)
                    img.save(self.local_path())
                    img.thumbnail(thumbnail_size)
                    img.save(self.local_thumbnail_path())
            elif self.definition.id == 'gray_detected_staffs':
                import json
                img = Image.open(File(self.page, 'gray_deskewed').local_path())
                with open(File(self.page, 'detected_staffs').local_path(), 'r') as f:
                    staffs = MusicLines.from_json(json.load(f))
                    img = np.array(img)
                    staffs.draw(img, color=(0, ))
                    img = Image.fromarray(img)
                    img.save(self.local_path())
                    img.thumbnail(thumbnail_size)
                    img.save(self.local_thumbnail_path())
            elif self.definition.id == 'binary_detected_staffs':
                import json
                img = Image.open(File(self.page, 'binary_deskewed').local_path())
                with open(File(self.page, 'detected_staffs').local_path(), 'r') as f:
                    staffs = MusicLines.from_json(json.load(f))
                    img = np.array(img)
                    staffs.draw(img, color=(0, ))
                    img = Image.fromarray(img)
                    img.save(self.local_path())
                    img.thumbnail(thumbnail_size)
                    img.save(self.local_thumbnail_path())
            elif self.definition.id == 'dewarped_gray' or self.definition.id == 'dewarped_binary' \
                    or self.definition.id == 'dewarped_original':
                from omr.dewarping.dummy_dewarper import dewarp
                import json
                orig, gray, binary = dewarp(
                    (Image.open(File(self.page, 'cropped_original').local_path()),
                     Image.open(File(self.page, 'cropped_gray').local_path()),
                     Image.open(File(self.page, 'cropped_binary').local_path())),
                    MusicLines.from_json(json.load(open(File(self.page, 'annotation').local_path(), 'r')))
                )
                orig.save(self.local_path(0))
                gray.save(self.local_path(1))
                binary.save(self.local_path(2))
            else:
                raise Exception("Cannot create file for {}".format(self.definition.id))









