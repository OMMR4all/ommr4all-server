from typing import NamedTuple, List
from PIL import Image
from multiprocessing import Lock
from locked_dict.locked_dict import LockedDict
import numpy as np
import os
from database.database_page import DatabasePage
import logging
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter

logger = logging.getLogger(__name__)


class DatabaseFileDefinition(NamedTuple):
    id: str
    output: List[str] = []
    requires: List[str] = []
    default: int = 0
    has_preview: bool = False
    has_hash_of_requirements: bool = False
    recalculate: bool = False


file_definitions = {
    'statistics': DatabaseFileDefinition(
        'statistics',
        ['statistics.json'],
    ),
    'statistics_backup': DatabaseFileDefinition(
        'statistics_backup',
        ['statistics_backup.zip'],
    ),
    'page_progress': DatabaseFileDefinition(
        'page_progress',
        ['page_progress.json'],
    ),
    'page_progress_backup': DatabaseFileDefinition(
        'page_progress_backup',
        ['page_progress_backup.zip'],
    ),
    'meta': DatabaseFileDefinition(
        'meta',
        ['meta.json'],
    ),
    'annotation': DatabaseFileDefinition(
        'annotation',
        ['annotation.json'],
        requires=['color_original'],
    ),
    'pcgts': DatabaseFileDefinition(
        'pcgts',
        ['pcgts.json'],
        requires=['color_original'],
    ),
    'monodiplus': DatabaseFileDefinition(
        'monodiplus',
        ['monodiplus.json'],
        requires=['pcgts'],
        recalculate=True,
    ),
    'monodiplus_svg': DatabaseFileDefinition(
        'monodiplus_svg',
        ['monodiplus.svg'],
        requires=['monodiplus'],
        recalculate=True,
    ),
    'pcgts_backup': DatabaseFileDefinition(
        'pcgts_backup',
        ['pcgts_backup.zip'],
    ),
    'color_original': DatabaseFileDefinition(
        'color_original',
        ['color_original.jpg'],
        has_preview=True,
    ),
    'color_highres_preproc': DatabaseFileDefinition(
        'color_highres_preproc',
        ['color_highres_preproc.jpg', 'gray_highres_preproc.jpg', 'binary_highres_preproc.png'],
        requires=['color_original'],
        has_preview=True,
    ),
    'gray_highres_preproc': DatabaseFileDefinition(
        'gray_highres_preproc',
        ['gray_highres_preproc.jpg'],
        requires=['color_highres_preproc'],
        has_preview=True,
    ),
    'binary_highres_preproc': DatabaseFileDefinition(
        'binary_highres_preproc',
        ['binary_highres_preproc.png'],
        requires=['color_highres_preproc'],
        has_preview=True,
    ),
    'color_lowres_preproc': DatabaseFileDefinition(
        'color_lowres_preproc',
        ['color_lowres_preproc.jpg', 'gray_lowres_preproc.jpg', 'binary_lowres_preproc.png'],
        requires=['color_highres_preproc'],
        has_preview=True,
    ),
    'gray_lowres_preproc': DatabaseFileDefinition(
        'gray_lowres_preproc',
        ['gray_lowres_preproc.jpg'],
        requires=['color_lowres_preproc'],
        has_preview=True,
    ),
    'binary_lowres_preproc': DatabaseFileDefinition(
        'binary_lowres_preproc',
        ['binary_lowres_preproc.png'],
        requires=['color_lowres_preproc'],
        has_preview=True,
    ),
    'color_norm': DatabaseFileDefinition(
        'color_norm',
        ['color_norm.jpg', 'gray_norm.jpg', 'binary_norm.png'],
        requires=['color_highres_preproc', 'color_original'],
        has_preview=True,
    ),
    'gray_norm': DatabaseFileDefinition(
        'gray_norm',
        ['gray_norm.jpg'],
        requires=['color_norm'],
        has_preview=True,
    ),
    'binary_norm': DatabaseFileDefinition(
        'binary_norm',
        ['binary_norm.png'],
        requires=['color_norm'],
        has_preview=True,
    ),
    'color_norm_x2': DatabaseFileDefinition(
        'color_norm_x2',
        ['color_norm_x2.jpg', 'gray_norm_x2.jpg', 'binary_norm_x2.png'],
        requires=['color_highres_preproc', 'color_original', 'color_norm'],
        has_preview=True,
    ),
    'gray_norm_x2': DatabaseFileDefinition(
        'gray_norm_x2',
        ['gray_norm_x2.jpg'],
        requires=['color_norm_x2'],
        has_preview=True,
    ),
    'binary_norm_x2': DatabaseFileDefinition(
        'binary_norm_x2',
        ['binary_norm_x2.png'],
        requires=['color_norm_x2'],
        has_preview=True,
    ),

    'connected_components_norm': DatabaseFileDefinition(
        'connected_components_norm',
        ['connected_components_norm.pkl'],
        requires=['binary_norm'],
    ),
}

mutex_dict = LockedDict()

thumbnail_size = (200, 350)

high_res_max_width = 2000
low_res_max_width = 1000
target_staff_line_distance = 10


class DatabaseFile:
    @staticmethod
    def file_definitions():
        return file_definitions

    def __init__(self, page: DatabasePage, fileId: str, create_if_not_existing=False):
        self.page = page
        self._fileId = fileId.strip('/')
        if self._fileId.endswith('_preview'):
            self.preview = True
            self.definition: DatabaseFileDefinition = file_definitions[self._fileId[:-len('_preview')]]
        else:
            self.preview = False
            self.definition: DatabaseFileDefinition = file_definitions[fileId.strip('/')]

        if create_if_not_existing and not self.exists() or create_if_not_existing and self.definition.recalculate:
            self.create()

    def local_path(self, file_id=-1):
        return os.path.join(self.page.local_path(), self.filename(file_id))

    def local_thumbnail_path(self, file_id=-1):
        return os.path.splitext(self.local_path(file_id))[0] + '_preview.jpg'

    def local_request_path(self):
        if self.preview:
            return self.local_thumbnail_path()
        else:
            return self.local_path()

    def filename(self, file_id=-1):
        return self.definition.output[file_id if file_id >= 0 else self.definition.default]

    def ext(self):
        return os.path.splitext(self.local_request_path())[-1]

    def remote_path(self):
        if self.preview:
            return os.path.join(self.page.remote_path(), self.definition.id + '_preview')
        else:
            return os.path.join(self.page.remote_path(), self.definition.id)

    def exists(self):
        return all(map(os.path.exists, [self.local_path(i) for i in range(len(self.definition.output))])) \
               and (not self.definition.has_preview or all(
            map(os.path.exists, [self.local_thumbnail_path(i) for i in range(len(self.definition.output))])))

    def delete(self):
        for i in range(len(self.definition.output)):
            if os.path.exists(self.local_path(file_id=i)):
                os.remove(self.local_path(file_id=i))
            if os.path.exists(self.local_thumbnail_path(file_id=i)):
                os.remove(self.local_thumbnail_path(file_id=i))

    def _save_and_thumbnail(self, img: Image, idx: int):
        img.save(self.local_path(idx))
        img.thumbnail(thumbnail_size)
        img.save(self.local_thumbnail_path(idx))

    def create(self):

        with mutex_dict.get(self.local_path(), Lock()):
            if self.exists() and not self.definition.recalculate:
                # check if exists
                return

            # check if requirement files exist
            for file in self.definition.requires:
                DatabaseFile(self.page, file).create()

            # check again if exists since the requirements might have created that file!
            if self.exists() and not self.definition.recalculate:
                return

            from omr.steps.preprocessing.preprocessing import Preprocessing

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
                from database.file_formats.pcgts import PcGts, Page, Meta
                img = Image.open(DatabaseFile(self.page, 'color_original').local_path())
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
            elif self.definition.id == 'color_original':
                # create preview
                img = Image.open(self.local_path())
                img.thumbnail(thumbnail_size)
                img.save(self.local_thumbnail_path())
            elif self.definition.id == 'color_highres_preproc':
                meta = self.page.meta()
                preproc = Preprocessing()
                img = Image.open(self.page.local_file_path('color_original.jpg'))
                w, h = img.size
                out_w = min(high_res_max_width, w)
                out_h = (out_w * h) // w
                c_hr = img.resize((out_w, out_h), Image.BILINEAR)
                c_hr, g_hr, b_hr = preproc.preprocess(c_hr)
                meta.preprocessing.deskewing_degrees = preproc.deskewed_angle
                meta.save(self.page)
                self._save_and_thumbnail(c_hr, 0)
                self._save_and_thumbnail(g_hr, 1)
                self._save_and_thumbnail(b_hr, 2)
            elif self.definition.id == 'color_lowres_preproc':
                c_hr = Image.open(self.page.local_file_path('color_highres_preproc.jpg'))
                b_hr = Image.open(self.page.local_file_path('binary_highres_preproc.png'))
                g_hr = Image.open(self.page.local_file_path('gray_highres_preproc.jpg'))
                w, h = c_hr.size
                out_w = min(low_res_max_width, w)
                out_h = (out_w * h) // w
                size = (out_w, out_h)
                c_hr = c_hr.resize(size, Image.BILINEAR)
                g_hr = g_hr.resize(size, Image.BILINEAR)
                b_hr = b_hr.resize(size, Image.NEAREST)

                self._save_and_thumbnail(c_hr, 0)
                self._save_and_thumbnail(g_hr, 1)
                self._save_and_thumbnail(b_hr, 2)
            elif self.definition.id == 'color_norm':
                meta = self.page.meta()
                c_hr = Image.open(self.page.local_file_path('color_highres_preproc.jpg'))
                if meta.preprocessing.auto_line_distance:
                    from omr.steps.preprocessing.scale.scale import LineDistanceComputer
                    ldc = LineDistanceComputer()
                    low_binary = Image.open(self.page.local_file_path('binary_highres_preproc.png'))
                    line_distance = ldc.get_line_distance(np.array(low_binary) / 255).line_distance
                    meta.preprocessing.average_line_distance = line_distance
                    meta.save(self.page)
                else:
                    # average_line_distance is expected to be computed on the original image
                    c_orig = Image.open(self.page.local_file_path('color_original.jpg'))
                    line_distance = int(
                        np.round(meta.preprocessing.average_line_distance * c_hr.size[0] / c_orig.size[0]))

                assert (line_distance > 0)

                # rescale original image
                scaling = line_distance / target_staff_line_distance
                size = (int(c_hr.size[0] / scaling), int(c_hr.size[1] / scaling))
                c_hr = c_hr.resize(size, Image.BILINEAR)

                # compute gray and binary based on normalized color image
                preproc = Preprocessing()
                g_hr = preproc.im2gray(c_hr)
                b_hr = preproc.binarize(c_hr)

                # save output
                self._save_and_thumbnail(c_hr, 0)
                self._save_and_thumbnail(g_hr, 1)
                self._save_and_thumbnail(b_hr, 2)
            elif self.definition.id == 'color_norm_x2':
                meta = self.page.meta()
                line_distance = meta.preprocessing.average_line_distance
                if line_distance <= 0:
                    nf = self.page.file('color_norm')
                    nf.delete()
                    nf.create()
                    meta = self.page.meta()
                    line_distance = meta.preprocessing.average_line_distance

                assert (line_distance > 0)
                c_hr = Image.open(self.page.local_file_path('color_highres_preproc.jpg'))

                # rescale original image
                scaling = line_distance / (target_staff_line_distance * 2)
                size = (int(c_hr.size[0] / scaling), int(c_hr.size[1] / scaling))
                c_hr = c_hr.resize(size, Image.BILINEAR)

                # compute gray and binary based on normalized color image
                preproc = Preprocessing()
                g_hr = preproc.im2gray(c_hr)
                b_hr = preproc.binarize(c_hr)

                # save output
                self._save_and_thumbnail(c_hr, 0)
                self._save_and_thumbnail(g_hr, 1)
                self._save_and_thumbnail(b_hr, 2)
            elif self.definition.id == 'connected_components_norm':
                import pickle
                from omr.steps.preprocessing.util.connected_compontents import connected_compontents_with_stats
                binary = np.array(Image.open(DatabaseFile(self.page, 'binary_norm').local_path()))
                with open(self.local_path(), 'wb') as f:
                    pickle.dump(connected_compontents_with_stats(binary), f)
            elif self.definition.id == 'monodiplus':
                import json
                import database.file_formats.pcgts as ns_pcgts
                # with open(DatabaseFile(self.page, 'pcgts').local_path()) as json_file:
                # pcgts = json.load(json_file)
                pcgts = ns_pcgts.PcGts.from_file(DatabaseFile(self.page, 'pcgts'))
                root = PcgtsToMonodiConverter([pcgts]).root
                # import hashlib
                with open(self.local_path(), 'w', encoding='utf-8') as f:
                    json.dump(root.to_json(), f, ensure_ascii=False, indent=4)
            elif self.definition.id == 'monodiplus_svg':
                path = DatabaseFile(self.page, 'monodiplus').local_path()
                from ommr4all.settings import BASE_DIR
                script_path = os.path.join(BASE_DIR, 'internal_storage', 'resources', 'monodi_svg_render', 'bin',
                                           'one-shot')
                import subprocess
                proc = subprocess.Popen([script_path, path, "-o", self.local_path()], stdout=subprocess.PIPE)
                result, err = proc.communicate()
                # error code in the java script is to be ignored for now
                exit_code = proc.wait()
            else:
                raise Exception("Cannot create file for {}".format(self.definition.id))


if __name__ == "__main__":
    from database import DatabaseBook
    import database.file_formats.pcgts as ns_pcgts

    b = DatabaseBook('demo')
    for b in b.pages():
        b.file('monodiplus_svg', create_if_not_existing=True)
