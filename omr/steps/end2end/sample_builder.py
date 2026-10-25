import io
import logging
import string
from typing import Collection, List, NamedTuple, Optional, Tuple

from PIL import Image

from database.file_formats.pcgts import PcGts, BlockType
from database.file_formats.pcgts.page import Block, Line, Page
from database.file_formats.pcgts.page.annotations import Connection
from omr.dataset import DatasetCallback
from omr.end2end.codec.data_generator import process_single_block

logger = logging.getLogger(__name__)

# Horizontal padding (in original image pixels) around the combined music+lyric crop.
# Must stay identical between training and prediction.
LEFT_PAD_PX = 50

CropBox = Tuple[float, float, float, float]


class Sample(NamedTuple):
    """Plain data only (no PcGts references), so that data loader workers receive it cheaply
    whatever their multiprocessing start method."""
    # the crop, scaled to the network input height and JPEG encoded to keep large datasets in memory
    image_jpeg: bytes
    gt: str
    # per symbol token of `gt`, in order: its position as a fraction (x, y) of the crop width and height
    symbol_locs: List[Tuple[float, float]]

    def image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_jpeg)).convert('RGB')


def find_text_region_for_music_block(page: Page, block: Block,
                                     connections: Optional[List[Connection]] = None) -> Optional[Block]:
    """Selects the lyric region belonging to a music block.

    With annotations (training) the candidates are the connected text regions; without
    (prediction) all lyric blocks below the top of the music block. Both use the same
    distance metric so training and prediction crops stay consistent.
    """
    if connections is not None:
        candidates = list(set(c.text_region for c in connections if c.text_region))
    else:
        candidates = [b for b in page.blocks_of_type([BlockType.LYRICS])
                      if b.aabb.top() >= block.aabb.top()]

    closest = None
    smallest_dist = None
    for tr in candidates:
        d = abs(tr.aabb.top() - block.aabb.bottom())
        if smallest_dist is None or d < smallest_dist:
            closest = tr
            smallest_dist = d
    return closest


def lyric_lines_for_music_block(page: Page, block: Block, region: Optional[Block]) -> List[Tuple[Block, Line]]:
    """(lyrics block, line) of every lyric line below `block`, left to right.

    A chant that starts mid-line splits the lyrics below one music line into two lines: two
    lyrics blocks (layout step) or two lines of one block (the editor's line splitter). The
    candidates are the lines in the row of `region`, the closest lyrics block that also defines
    the crop, that overlap the music block horizontally."""
    if region is None:
        return []
    row, music = region.aabb, block.aabb
    lines = [(b, l) for b in page.blocks_of_type([BlockType.LYRICS]) for l in b.lines
             if l.aabb.bottom() > row.top() and l.aabb.top() < row.bottom()
             and l.aabb.right() > music.left() and l.aabb.left() < music.right()]
    return sorted(lines, key=lambda bl: bl[1].aabb.left())


def compute_crop_box(page: Page, block: Block, text_region: Optional[Block]) -> CropBox:
    """(left, top, right, bottom) of the crop in original image pixels."""
    combined_aabb = block.aabb
    top_pad = 0
    if text_region is not None:
        combined_aabb = combined_aabb.union(text_region.aabb)
        top_pad = 1 / 8 * text_region.aabb.height()

    return (
        page.page_to_image_scale(combined_aabb.left()) - LEFT_PAD_PX,
        page.page_to_image_scale(combined_aabb.top() - top_pad),
        page.page_to_image_scale(combined_aabb.right()) + LEFT_PAD_PX,
        page.page_to_image_scale(combined_aabb.bottom()),
    )


def crop_block_image(full_image: Image.Image, page: Page, block: Block,
                     text_region: Optional[Block]) -> Image.Image:
    return full_image.crop(compute_crop_box(page, block, text_region))


def page_point_to_crop_fraction(page: Page, box: CropBox, x: float, y: float) -> Tuple[float, float]:
    left, top, right, bottom = box
    return (float((page.page_to_image_scale(x) - left) / (right - left)),
            float((page.page_to_image_scale(y) - top) / (bottom - top)))


def crop_fraction_to_page_point(page: Page, box: CropBox, fx: float, fy: float) -> Tuple[float, float]:
    left, top, right, bottom = box
    return (page.image_to_page_scale(left + fx * (right - left)),
            page.image_to_page_scale(top + fy * (bottom - top)))


def collect_unique_chars(pcgts_list: List[PcGts]) -> List[str]:
    chars = [string.ascii_lowercase + string.ascii_uppercase + string.digits]
    for p in pcgts_list:
        for tl in p.page.all_text_lines():
            chars.append(tl.text())
    return sorted(set("".join(chars)))


def scale_to_height(image: Image.Image, height: int) -> Image.Image:
    w, h = image.size
    return image.resize((max(1, round(w * height / h)), height), Image.BICUBIC)


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def build_training_samples(pcgts_list: List[PcGts],
                           target_height: int,
                           class_ids: Collection[str] = (),
                           callback: Optional[DatasetCallback] = None,
                           melody: bool = False) -> List[Sample]:
    samples = []
    if callback:
        callback.loading_started(len(pcgts_list))

    for i, p in enumerate(pcgts_list):
        page = p.page
        try:
            full_image = Image.open(p.dataset_page().file("color_original").local_path())
            full_image.load()
        except Exception as e:
            logger.warning(f"Skipping page {page.location.page}: could not open image ({e})")
            continue

        for block in page.blocks_of_type([BlockType.MUSIC]):
            connections = [c for c in page.annotations.connections if c.music_region == block]
            if not connections:
                continue
            text_region = find_text_region_for_music_block(page, block, connections)
            if text_region is None:
                continue

            transcription = process_single_block(page, block, connections, class_ids, melody=melody)
            if not transcription:
                continue

            box = compute_crop_box(page, block, text_region)
            locs = [page_point_to_crop_fraction(page, box, s.coord.x, s.coord.y) for s in transcription.symbols]
            image = scale_to_height(full_image.crop(box), target_height)
            samples.append(Sample(image_jpeg=encode_jpeg(image), gt=transcription.text, symbol_locs=locs))

        if callback:
            callback.loading(i + 1, len(pcgts_list))

    if callback:
        callback.loading_finished(len(pcgts_list))
    return samples
