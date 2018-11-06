from . import *
from omr.datatypes.page import TextRegion, MusicRegion
from omr.datatypes.page import annotations as annotations
from typing import List
import numpy as np
from PIL import Image
import os
import main.book as book


class Page:
    def __init__(self,
                 text_regions: List[TextRegion]=None,
                 music_regions: List[MusicRegion]=None,
                 image_filename="", image_height=0, image_width=0,
                 location: book.Page = None):
        self.text_regions = text_regions if text_regions else []
        self.music_regions = music_regions if music_regions else []
        self.image_filename = image_filename
        self.image_height = image_height
        self.image_width = image_width
        self.annotations = annotations.Annotations(self)
        self.location = location
        if not os.path.exists(location.local_file_path(self.image_filename)):
            raise Exception("Image not found at {}".format(location.local_file_path(self.image_filename)))

    def syllable_by_id(self, syllable_id):
        for t in self.text_regions:
            r = t.syllable_by_id(syllable_id)
            if r:
                return r

        return None

    def _resolve_cross_refs(self):
        for t in self.text_regions:
            t._resolve_cross_refs(self)

    @staticmethod
    def from_json(json: dict, location: book.Page):
        page = Page(
            [TextRegion.from_json(t) for t in json.get('textRegions', [])],
            [MusicRegion.from_json(m) for m in json.get('musicRegions', [])],
            json.get('imageFilename', ""),
            json.get('imageHeight', 0),
            json.get('imageWidth', 0),
            location=location,
        )
        if 'annotations' in json:
            page.annotations = annotations.Annotations.from_json(json['annotations'], page)
        page._resolve_cross_refs()
        return page

    def to_json(self):
        return {
            "textRegions": [t.to_json() for t in self.text_regions],
            "musicRegions": [m.to_json() for m in self.music_regions],
            "imageFilename": self.image_filename,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            'annotations': self.annotations.to_json(),
        }

    def music_region_by_id(self, id: str):
        for mr in self.music_regions:
            if mr.id == id:
                return mr
        return None

    def text_region_by_id(self, id: str):
        for tr in self.text_regions:
            if tr.id == id:
                return tr
        return None

    def staff_equivs(self, index):
        return [m.staff_equiv_by_index(index) for m in self.music_regions if m.has_staff_equiv_by_index(index)]

    def avg_staff_distance(self, index):
        staffs = self.staff_equivs(index)
        d = []
        for i in range(1, len(staffs)):
            top = staffs[i - 1].staff_lines[-1].center_y()
            bot = staffs[i].staff_lines[0].center_y()
            d.append(bot - top)

        return np.mean(d)

    def avg_staff_line_distance(self, index):
        staffs = self.staff_equivs(index)
        avg = np.mean([v for v in [d.avg_staff_line_distance for d in staffs] if v > 0])
        return max([5, avg])

    def draw(self, index, canvas, color=(0, 255, 0), thickness=-1):
        avg = self.avg_staff_line_distance(index)

        if thickness < 0:
            thickness = avg / 10 if avg > 0 else 5

        for staff in self.staff_equivs(index):
            staff.draw(canvas, color, thickness)

    def extract_music_line_images_and_gt(self, dewarped=True):
        pass

