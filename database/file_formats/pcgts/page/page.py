from database.file_formats.pcgts.page import Coords, Point, Block, BlockType, Line, Rect, Size
from database.file_formats.pcgts.page import annotations as annotations
from database.file_formats.pcgts.page.usercomment import UserComments
from database.file_formats.pcgts.page.readingorder import ReadingOrder
from typing import List, TYPE_CHECKING, Union, Optional, Iterable
import numpy as np
from enum import IntEnum

if TYPE_CHECKING:
    from database import DatabasePage


class PageScaleReference(IntEnum):
    ORIGINAL = 0
    HIGHRES = 1
    LOWRES = 2
    NORMALIZED = 3
    NORMALIZED_X2 = 4

    def file(self, color: str = "color"):
        if self.value == PageScaleReference.ORIGINAL:
            return color + "_original"
        elif self.value == PageScaleReference.HIGHRES:
            return color + "_highres_preproc"
        elif self.value == PageScaleReference.LOWRES:
            return color + "_lowres_preproc"
        elif self.value == PageScaleReference.NORMALIZED:
            return color + "_norm"
        elif self.value == PageScaleReference.NORMALIZED_X2:
            return color + "_norm_x2"


class Page:
    def __init__(self,
                 blocks: List[Block]=None,
                 image_filename="", image_height=0, image_width=0,
                 location: 'DatabasePage' = None):
        self.blocks: List[Block] = blocks if blocks else []
        self.image_filename = image_filename
        self.image_height = image_height
        self.image_width = image_width
        self.annotations = annotations.Annotations(self)
        self.comments = UserComments(self)
        self.reading_order = ReadingOrder(self)
        self.location = location
        self.page_scale_ratios = {}

        self.update_note_names()

    def syllable_by_id(self, syllable_id):
        for b in self.blocks:
            r = b.syllable_by_id(syllable_id)
            if r:
                return r

        return None

    @staticmethod
    def from_json(json: dict, location: Optional['DatabasePage']):
        page = Page(
            [Block.from_json(t) for t in json.get('blocks', [])],
            json.get('imageFilename', ""),
            json.get('imageHeight', 0),
            json.get('imageWidth', 0),
            location=location,
        )
        if 'annotations' in json:
            page.annotations = annotations.Annotations.from_json(json['annotations'], page)

        if 'comments' in json:
            page.comments = UserComments.from_json(json['comments'], page)

        if 'readingOrder' in json:
            page.reading_order = ReadingOrder.from_json(json['readingOrder'], page)

        page.update_note_names()
        return page

    def to_json(self):
        return {
            "blocks": [t.to_json() for t in self.blocks],
            "imageFilename": self.image_filename,
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            'annotations': self.annotations.to_json(),
            'comments': self.comments.to_json(),
            'readingOrder': self.reading_order.to_json(),
        }

    def blocks_of_type(self, block_type: Union[BlockType, Iterable[BlockType]]) -> List[Block]:
        try:
            block_types = list(block_type)
            return [b for b in self.blocks if b.block_type in block_types]
        except TypeError:
            return [b for b in self.blocks if b.block_type == block_type]

    def clear_blocks_of_type(self, block_type: BlockType):
        self.blocks = [b for b in self.blocks if b.block_type != block_type]

    def clear_text_blocks(self):
        for b in self.blocks:
            if b.block_type == BlockType.MUSIC:
                continue

            self.annotations.drop_annotation_by_text_block(b)

        self.blocks = [b for b in self.blocks if b.block_type == BlockType.MUSIC]

    def music_blocks(self):
        return self.blocks_of_type(BlockType.MUSIC)

    def block_of_line(self, l: Line):
        for b in self.blocks:
            if l in b.lines:
                return b

        return None

    def text_blocks(self):
        return [b for b in self.blocks if b.block_type != BlockType.MUSIC]

    def block_by_id(self, id: str) -> Optional[Block]:
        for b in self.blocks:
            if b.id == id:
                return b
        return None

    def line_by_id(self, id: str) -> Optional[Line]:
        for b in self.blocks:
            l = b.line_by_id(id)
            if l:
                return l
        return None

    def music_region_by_id(self, id: str) -> Optional[Block]:
        for mr in self.blocks_of_type(BlockType.MUSIC):
            if mr.id == id:
                return mr
        return None

    def music_line_by_id(self, id: str) -> Optional[Line]:
        for mr in self.blocks_of_type(BlockType.MUSIC):
            l = mr.line_by_id(id)
            if l:
                return l

        return None

    def text_region_by_id(self, id: str) -> Optional[Block]:
        for tr in self.text_blocks():
            if tr.id == id:
                return tr
        return None

    def text_line_by_id(self, id: str) -> Optional[Line]:
        for tr in self.text_blocks():
            l = tr.line_by_id(id)
            if l:
                return l

        return None

    def all_music_lines(self) -> List[Line]:
        return sum([mr.lines for mr in self.music_blocks()], [])

    def all_text_lines(self):
        return sum([b.lines for b in self.text_blocks()], [])

    def all_staves_staff_line_coords(self, scale: Optional[PageScaleReference] = None) -> List[List[Coords]]:
        staves: List[List[Coords]] = []
        for mr in self.music_blocks():
            if scale:
                staves += [[self.page_to_image_scale(sl.coords, scale) for sl in ml.staff_lines] for ml in mr.lines]
            else:
                staves += [[sl.coords for sl in ml.staff_lines] for ml in mr.lines]

        return staves

    def avg_staff_line_distance(self):
        staffs = self.all_music_lines()
        avg = np.mean([v for v in [d.avg_line_distance(default=-1) for d in staffs] if v > 0])
        return max([0.001, avg])

    def closest_music_line_to_text_line(self, tl: Line):
        closest = None
        d = 10000000000
        for ml in self.all_music_lines():
            dp = tl.aabb.top() - ml.aabb.top()
            if dp < 0:
                continue
            elif d > dp:
                d = dp
                closest = ml

        return closest

    def draw(self, canvas, color=(0, 255, 0), thickness=-1):
        avg = self.avg_staff_line_distance()

        if thickness < 0:
            thickness = avg / 10 if avg > 0 else 5

        for staff in self.all_music_lines():
            staff.draw(canvas, color, thickness)

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
        elif isinstance(p, Size):
            return p.scale(scale)
        elif isinstance(p, Rect):
            return Rect(self._scale(p.origin, scale), self._scale(p.size, scale))
        elif isinstance(p, Iterable):
            return np.array(p) * scale
        else:
            return p * scale

    def image_to_page_scale(self, p: Union[Coords, Point, float, int], ref: PageScaleReference = PageScaleReference.ORIGINAL):
        return self._scale(p, 1.0 / self.page_scale_size(ref)[1])

    def page_to_image_scale(self, p: Union[Coords, Point, float, int], ref: PageScaleReference = PageScaleReference.ORIGINAL):
        return self._scale(p, self.page_scale_size(ref)[1])

    def rotate(self, degrees: float):
        for block in self.blocks:
            block.rotate(degrees, origin=(0.5 * self.image_width / self.image_height, 0.5))

    def sort_blocks(self):
        self.blocks.sort(key=lambda block: block.aabb.top())

    def update_note_names(self):
        current_clef = None
        for b in sorted(self.music_blocks(), key=lambda b: b.aabb.top()):
            current_clef = b.update_note_names(current_clef)

