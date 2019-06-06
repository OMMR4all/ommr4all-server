from database.file_formats.pcgts.page import TextRegion, MusicRegion, Coords, Point
from database.file_formats.pcgts.page import annotations as annotations
from database.file_formats.pcgts.page.usercomment import UserComments
from database.file_formats.pcgts.page.readingorder import ReadingOrder
from typing import List, TYPE_CHECKING, Union, Optional
import numpy as np
from enum import IntEnum

if TYPE_CHECKING:
    from database import DatabasePage


class PageScaleReference(IntEnum):
    ORIGINAL = 0
    HIGHRES = 1
    LOWRES = 2
    NORMALIZED = 3

    def file(self, color: str = "color"):
        if self.value == PageScaleReference.ORIGINAL:
            return color + "_original"
        elif self.value == PageScaleReference.HIGHRES:
            return color + "_highres_preproc"
        elif self.value == PageScaleReference.LOWRES:
            return color + "_lowres_preproc"
        elif self.value == PageScaleReference.NORMALIZED:
            return color + "_norm"


class Page:
    def __init__(self,
                 text_regions: List[TextRegion]=None,
                 music_regions: List[MusicRegion]=None,
                 image_filename="", image_height=0, image_width=0, relative_coords=False,
                 location: 'DatabasePage' = None):
        self.text_regions: List[TextRegion] = text_regions if text_regions else []
        self.music_regions: List[MusicRegion] = music_regions if music_regions else []
        self.image_filename = image_filename
        self.image_height = image_height
        self.image_width = image_width
        self.annotations = annotations.Annotations(self)
        self.comments = UserComments(self)
        self.reading_order = ReadingOrder(self)
        self.location = location
        self.relative_coords = relative_coords
        self.page_scale_ratios = {}

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
    def from_json(json: dict, location: 'DatabasePage'):
        page = Page(
            [TextRegion.from_json(t) for t in json.get('textRegions', [])],
            [MusicRegion.from_json(m) for m in json.get('musicRegions', [])],
            json.get('imageFilename', ""),
            json.get('imageHeight', 0),
            json.get('imageWidth', 0),
            json.get('relativeCoords', False),
            location=location,
        )
        if 'annotations' in json:
            page.annotations = annotations.Annotations.from_json(json['annotations'], page)

        if 'comments' in json:
            page.comments = UserComments.from_json(json['comments'], page)

        if 'readingOrder' in json:
            page.reading_order = ReadingOrder.from_json(json['readingOrder'], page)

        page._resolve_cross_refs()
        return page

    def to_json(self):
        return {
            "textRegions": [t.to_json() for t in self.text_regions],
            "musicRegions": [m.to_json() for m in self.music_regions],
            "imageFilename": self.image_filename,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            "relativeCoords": self.relative_coords,
            'annotations': self.annotations.to_json(),
            'comments': self.comments.to_json(),
            'readingOrder': self.reading_order.to_json(),
        }

    def music_region_by_id(self, id: str):
        for mr in self.music_regions:
            if mr.id == id:
                return mr
        return None

    def music_line_by_id(self, id: str):
        for mr in self.music_regions:
            for ml in mr.staffs:
                if ml.id == id:
                    return ml

        return None

    def text_region_by_id(self, id: str):
        for tr in self.text_regions:
            if tr.id == id:
                return tr
        return None

    def text_line_by_id(self, id: str):
        for tr in self.text_regions:
            for tl in tr.text_lines:
                if tl.id == id:
                    return tl

        return None

    def all_music_lines(self):
        return [ml for mr in self.music_regions for ml in mr.staffs]

    def all_text_lines(self):
        return [tl for tr in self.text_regions for tl in tr.text_lines]

    def all_staves_staff_line_coords(self, scale: Optional[PageScaleReference] = None) -> List[List[Coords]]:
        staves: List[List[Coords]] = []
        for mr in self.music_regions:
            if scale:
                staves += [[self.page_to_image_scale(sl.coords, scale) for sl in ml.staff_lines] for ml in mr.staffs]
            else:
                staves += [[sl.coords for sl in ml.staff_lines] for ml in mr.staffs]

        return staves

    def avg_staff_line_distance(self):
        staffs = self.all_music_lines()
        avg = np.mean([v for v in [d.avg_line_distance(default=-1) for d in staffs] if v > 0])
        return max([5, avg])

    def draw(self, canvas, color=(0, 255, 0), thickness=-1):
        avg = self.avg_staff_line_distance()

        if thickness < 0:
            thickness = avg / 10 if avg > 0 else 5

        for staff in self.all_music_lines():
            staff.draw(canvas, color, thickness)

    def extract_music_line_images_and_gt(self, dewarped=True):
        pass

    def page_scale_size(self, ref: PageScaleReference):
        if ref not in self.page_scale_ratios:
            from PIL import Image
            img = Image.open(self.location.file(ref.file(), create_if_not_existing=True).local_path())
            self.page_scale_ratios[ref] = img.size

        return self.page_scale_ratios[ref]

    def _scale(self, p: Union[Coords, Point, float, int], scale: float):
        if isinstance(p, Coords):
            return p.scale(scale)
        elif isinstance(p, Point):
            return p.scale(scale)
        else:
            return p * scale

    def image_to_page_scale(self, p: Union[Coords, Point, float, int], ref: PageScaleReference = PageScaleReference.ORIGINAL):
        return self._scale(p, 1.0 / self.page_scale_size(ref)[1])

    def page_to_image_scale(self, p: Union[Coords, Point, float, int], ref: PageScaleReference = PageScaleReference.ORIGINAL):
        return self._scale(p, self.page_scale_size(ref)[1])

    def to_relative_coords(self):
        if self.relative_coords:
            # already in relative coords
            return

        from .musicregion import Neume

        def i2p(p):
            # validity check
            if isinstance(p, Coords):
                assert(0.0 <= p.points.all() <= 1.0)
            elif isinstance(p, Point):
                assert(0.0 <= p.p.all() <= 1.0)
            else:
                assert(0.0 <= p <= 1.0)

            return self.image_to_page_scale(p, ref=PageScaleReference.ORIGINAL)

        for r in self.text_regions:
            r.coords = i2p(r.coords)
            for l in r.text_lines:
                l.coords = i2p(l.coords)

        for r in self.music_regions:
            r.coords = i2p(r.coords)
            for l in r.staffs:
                l.coords = i2p(l.coords)
                for s in l.staff_lines:
                    s.coords = i2p(s.coords)

                for s in l.symbols:
                    s.coord = i2p(s.coord)
                    if isinstance(s, Neume):
                        for nc in s.notes:
                            nc.coord = i2p(nc.coord)

        self.relative_coords = True
