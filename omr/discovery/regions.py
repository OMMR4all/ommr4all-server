"""Staff crops and the coordinate mapping between a crop and the page.

The crop pipeline is not reimplemented here. `SymbolDetectionDatasetTorch` already extracts
one `color_norm_x2` crop per music line and exposes the invertible coordinate chain that
`omr/steps/symboldetection/yolo_detector/predictor.py` uses to put a detection back onto the
page; this module wraps it and adds what discovery needs on top: a pixel aligned ink mask, the
staff lines in crop pixels and the staff space in crop pixels.

Two properties are load bearing:

* `CropConfig.dewarp` is False. `ImageExtractDewarpedStaffLineImages.global_to_local_pos`
  applies the *same* forward transform as `local_to_global_pos` rather than its inverse, so
  only without dewarping is the page -> crop direction correct. Verified by
  `tests/test_discovery_geometry.py`.
* `CropConfig.center` is False. `ImageExtractDewarpedStaffLineImages._resize_to_height` pads
  with a two dimensional block (`np.full((n, width), ...)`), which cannot be stacked onto the
  three channel colour crop this dataset produces -- the line would raise and be silently
  dropped by `Dataset._load`. The vertical extent is already normalised by the dataset (the
  staff line bounding box plus one average line distance above and below).
"""
import logging
from typing import Any, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

from database import DatabasePage
from database.file_formats.pcgts import BlockType, Line, Page, PageScaleReference, PcGts, Point
from omr.dataset import DatasetParams
from omr.discovery.config import CropConfig
from omr.steps.symboldetection.dataset import SymbolDetectionDatasetTorch

logger = logging.getLogger(__name__)


class StaffCrop(NamedTuple):
    """One music line as an image plus everything needed to map back to the page."""
    book: str
    page: str
    block_id: str
    line_id: str
    #: RGB uint8 crop taken from `color_norm_x2`, rescaled to `CropConfig.crop_height`
    image: np.ndarray
    #: ink mask of `image`, True where there is ink: Otsu on the crop's own grayscale,
    #: restricted to `region_mask`
    binary: np.ndarray
    #: which pixels of the crop may carry a music symbol at all, see `_region_mask`
    region_mask: np.ndarray
    #: one (K, 2) array of xy positions in crop pixels per staff line, top to bottom
    staff_lines_px: List[np.ndarray]
    #: one staff space in crop pixels
    staff_space_px: float
    #: one staff space in height-normalised page units
    staff_space_page: float
    line: Line
    page_obj: Page
    #: internals of the coordinate chain, used by `crop_box_to_page`/`page_point_to_crop`
    dataset: Any
    op_params: List[Any]
    scale_reference: PageScaleReference


def crop_dataset_params(cfg: CropConfig) -> DatasetParams:
    """A fresh `DatasetParams` for every dataset.

    `Dataset.__init__` -> `create_image_operation_list` mutates the params object (it sets
    `image_input` and `page_scale_reference`) and `Dataset.load` memoises its result, so a
    params instance must never be shared between datasets.
    """
    return DatasetParams(
        pad=list(cfg.pad),
        pad_power_of_2=None,
        dewarp=cfg.dewarp,
        center=cfg.center,
        staff_lines_only=cfg.staff_lines_only,
        cut_region=False,
        height=cfg.crop_height,
        gt_required=False,
        symbol_label_sets=None,
    )


def count_music_lines(pages: List[DatabasePage]) -> int:
    total = 0
    for page in pages:
        for block in page.pcgts().page.music_blocks():
            total += len(block.lines)
    return total


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Otsu binarisation of the crop itself.

    Deriving the mask from the same pixels that carry the features guarantees that mask and
    image are aligned; loading `binary_norm_x2.png` through a second dataset would not be,
    because the rescale to `crop_height` is applied to each dataset separately.
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary > 0


def _page_point_to_crop(page_obj: Page, dataset, op_params, ref: PageScaleReference,
                        x: float, y: float) -> Tuple[float, float]:
    p = dataset.global_to_local_pos(page_obj.page_to_image_scale(Point(x, y), ref), op_params)
    return float(p.x), float(p.y)


def _fill_polygon(canvas: np.ndarray, coords, to_crop) -> None:
    import cv2
    points = getattr(coords, 'points', None)
    if points is None or len(points) < 3:
        return
    mapped = np.array([to_crop(p[0], p[1]) for p in np.asarray(points, dtype=float)])
    cv2.fillPoly(canvas, [np.round(mapped).astype(np.int32)], 1)


def _region_mask(page_obj: Page, line: Line, shape: Tuple[int, int], staff_space_px: float,
                 cfg: CropConfig, to_crop) -> np.ndarray:
    """Which pixels of a crop may carry a music symbol.

    A crop reaches one average line distance below the lowest staff line, which on a densely
    written page already contains the lyrics line -- its ink would otherwise dominate the
    candidate set. The annotated layout answers this directly: keep the music line's own
    polygon (dilated, because a symbol may stick out of it) and drop everything that a
    non-music block claims.
    """
    import cv2
    allowed = np.ones(shape, dtype=bool)

    if cfg.restrict_to_line_polygon:
        inside = np.zeros(shape, dtype=np.uint8)
        _fill_polygon(inside, line.coords, to_crop)
        if inside.any():
            dilate = max(1, int(round(staff_space_px * cfg.line_polygon_dilate_staff_space)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            allowed &= cv2.dilate(inside, kernel) > 0

    if cfg.mask_non_music_blocks:
        blocked = np.zeros(shape, dtype=np.uint8)
        for block in page_obj.blocks:
            if block.block_type == BlockType.MUSIC:
                continue
            _fill_polygon(blocked, block.coords, to_crop)
            for other in block.lines:
                _fill_polygon(blocked, other.coords, to_crop)
        allowed &= blocked == 0
    return allowed


def page_point_to_crop(crop: StaffCrop, x: float, y: float) -> Tuple[float, float]:
    """Height-normalised page coordinates -> crop pixels."""
    return _page_point_to_crop(crop.page_obj, crop.dataset, crop.op_params, crop.scale_reference, x, y)


def crop_point_to_page(crop: StaffCrop, x: float, y: float) -> Tuple[float, float]:
    """Crop pixels -> height-normalised page coordinates."""
    p = crop.dataset.local_to_global_pos(Point(x, y), crop.op_params)
    p = crop.page_obj.image_to_page_scale(p, crop.scale_reference)
    return float(p.x), float(p.y)


def crop_box_to_page(crop: StaffCrop, x: float, y: float, w: float, h: float):
    """A box in crop pixels -> a `Box` in height-normalised page coordinates."""
    from omr.discovery.schema import Box
    x0, y0 = crop_point_to_page(crop, x, y)
    x1, y1 = crop_point_to_page(crop, x + w, y + h)
    return Box(x=min(x0, x1), y=min(y0, y1), w=abs(x1 - x0), h=abs(y1 - y0))


def page_box_to_crop(crop: StaffCrop, box) -> Tuple[float, float, float, float]:
    """A `Box` in height-normalised page coordinates -> (x, y, w, h) in crop pixels."""
    x0, y0 = page_point_to_crop(crop, box.x, box.y)
    x1, y1 = page_point_to_crop(crop, box.right(), box.bottom())
    return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def _staff_space_px(page_obj: Page, dataset, op_params, ref: PageScaleReference,
                    staff_space_page: float, fallback: float) -> float:
    """The vertical scale of the (affine, because dewarp is off) crop transform."""
    if staff_space_page <= 0:
        return fallback
    _, y0 = _page_point_to_crop(page_obj, dataset, op_params, ref, 0.0, 0.0)
    _, y1 = _page_point_to_crop(page_obj, dataset, op_params, ref, 0.0, staff_space_page)
    value = abs(y1 - y0)
    return value if value > 1e-6 else fallback


def staff_crops_of_page(page: DatabasePage, cfg: CropConfig) -> List[StaffCrop]:
    """Every music line of one page as a `StaffCrop`.

    Lines that the dataset drops (no staff lines, a broken image operation) are simply absent;
    the caller compares against `count_music_lines` to report them.
    """
    if cfg.dewarp:
        raise ValueError('CropConfig.dewarp must be False: the page -> crop direction of '
                         'ImageExtractDewarpedStaffLineImages is not the inverse of crop -> page')
    pcgts: PcGts = page.pcgts()
    dataset = SymbolDetectionDatasetTorch([pcgts], crop_dataset_params(cfg))
    page_obj = pcgts.page
    fallback_space_px = cfg.crop_height / 5.0

    out: List[StaffCrop] = []
    for data in dataset.load():
        line: Optional[Line] = data.operation.music_line
        block = data.operation.music_region
        if line is None or block is None:
            logger.warning('Skipping a crop of page %s without an associated line', page.page)
            continue
        image = np.ascontiguousarray(data.region)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        op_params = data.operation.params
        ref = data.operation.scale_reference

        staff_space_page = line.avg_line_distance(default=-1)
        if staff_space_page <= 0:
            staff_space_page = page_obj.avg_staff_line_distance()
        space_px = _staff_space_px(page_obj, dataset, op_params, ref, staff_space_page, fallback_space_px)

        staff_lines_px = []
        for staff_line in line.staff_lines.sorted():
            points = np.asarray(staff_line.coords.points, dtype=float)
            if points.size == 0:
                continue
            mapped = np.array([_page_point_to_crop(page_obj, dataset, op_params, ref, p[0], p[1])
                               for p in points], dtype=float)
            staff_lines_px.append(mapped)

        def to_crop(x, y, _page=page_obj, _ds=dataset, _p=op_params, _r=ref):
            return _page_point_to_crop(_page, _ds, _p, _r, x, y)

        region_mask = _region_mask(page_obj, line, image.shape[:2], float(space_px), cfg, to_crop)
        out.append(StaffCrop(
            book=page.book.book,
            page=page.page,
            block_id=block.id,
            line_id=line.id,
            image=image,
            binary=_ink_mask(image) & region_mask,
            region_mask=region_mask,
            staff_lines_px=staff_lines_px,
            staff_space_px=float(space_px),
            staff_space_page=float(staff_space_page),
            line=line,
            page_obj=page_obj,
            dataset=dataset,
            op_params=op_params,
            scale_reference=ref,
        ))
    return out


def iter_staff_crops(pages: List[DatabasePage], cfg: CropConfig) -> Iterator[StaffCrop]:
    for page in pages:
        for crop in staff_crops_of_page(page, cfg):
            yield crop
