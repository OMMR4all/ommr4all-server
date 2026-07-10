import logging
import string
from typing import List, NamedTuple, Optional, Tuple

from PIL import Image

from database.file_formats.pcgts import PcGts, BlockType
from database.file_formats.pcgts.page import Block, Page
from database.file_formats.pcgts.page.annotations import Connection
from omr.dataset import DatasetCallback
from omr.end2end.codec.data_generator import process_single_block

logger = logging.getLogger(__name__)

# Horizontal padding (in original image pixels) around the combined music+lyric crop.
# Must stay identical between training and prediction.
LEFT_PAD_PX = 50


class Sample(NamedTuple):
    image: Image.Image
    gt: Optional[str]
    pcgts: PcGts
    block: Block
    text_region: Optional[Block]


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


def compute_crop_box(page: Page, block: Block, text_region: Optional[Block]) -> Tuple[float, float, float, float]:
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


def collect_unique_chars(pcgts_list: List[PcGts]) -> List[str]:
    chars = [string.ascii_lowercase + string.ascii_uppercase + string.digits]
    for p in pcgts_list:
        for tl in p.page.all_text_lines():
            chars.append(tl.text())
    return sorted(set("".join(chars)))


def build_training_samples(pcgts_list: List[PcGts],
                           callback: Optional[DatasetCallback] = None,
                           melody: bool = False) -> List[Sample]:
    samples = []
    if callback:
        callback.loading_started(len(pcgts_list))

    for i, p in enumerate(pcgts_list):
        page = p.page
        try:
            full_image = Image.open(p.dataset_page().file("color_original").local_path())
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

            gt = process_single_block(page, block, connections,
                                      LEFT_PAD_PX, 1 / 8 * text_region.aabb.height(), melody=melody)
            if not gt:
                continue

            image = crop_block_image(full_image, page, block, text_region)
            samples.append(Sample(image=image, gt=gt, pcgts=p, block=block, text_region=text_region))

        if callback:
            callback.loading(i + 1, len(pcgts_list))

    if callback:
        callback.loading_finished(len(pcgts_list))
    return samples
