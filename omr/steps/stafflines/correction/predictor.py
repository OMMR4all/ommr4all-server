import logging
import os
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from database import DatabasePage
from database.file_formats.pcgts import Coords, Line, PageScaleReference, Point, SymbolType
from omr.steps.algorithm import AlgorithmPredictionResult, AlgorithmPredictionResultGenerator, AlgorithmPredictor, \
    AlgorithmPredictorSettings, FailedPageResult, PredictionCallback, PredictionProgress
from omr.steps.stafflines.correction.meta import Meta

logger = logging.getLogger(__name__)

# The image the fit runs on. NORMALIZED_X2 is scaled to 20 px per staff line distance, so
# one pixel of shift is 1/20 of a staff space -- fine enough that a sub-pixel refinement
# would not buy anything, while the search stays a handful of array lookups per stave.
SCALE_REFERENCE = PageScaleReference.NORMALIZED_X2

# Fraction of the score range within which a candidate shift still counts as "as good as
# the best one". Among those the smallest shift wins, so a stave that is already correct is
# not nudged around by image noise.
SCORE_TOLERANCE = 0.05

# Minimum spread (in gray levels) between the best and the worst candidate. Below that the
# stave sits on a uniform area -- e.g. a bleached or cropped-off staff -- and any minimum
# would be noise, so the stave is left alone.
MIN_CONTRAST = 1.0


class LineShift(NamedTuple):
    """The vertical correction found for one stave."""
    line_id: str
    dy: float          # in page (height normalised) coordinates, what gets applied
    dy_px: int         # the same shift in pixels of SCALE_REFERENCE, for logging/debugging
    score_before: float
    score_after: float


class PageShifts(NamedTuple):
    page: DatabasePage
    shifts: List[LineShift]


def _line_samples(coords: Coords, width: int, height: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Sample one staff line polyline at every image column it spans.

    Returns the integer x columns and the interpolated (float) y of the line at those
    columns, or None if the line does not cover at least two columns of the image.
    """
    points = coords.points
    if len(points) < 2:
        return None

    # interpolate_y() relies on np.interp, which expects the x of the polyline to increase
    order = np.argsort(points[:, 0])
    px, py = points[order, 0], points[order, 1]

    left = max(int(np.ceil(px[0])), 0)
    right = min(int(np.floor(px[-1])), width - 1)
    if right <= left:
        return None

    xs = np.arange(left, right + 1)
    ys = np.interp(xs, px, py)
    # Staves that stick out of the image would otherwise pull the score towards the clipped
    # border row, which is constant and therefore indistinguishable for every candidate.
    inside = (ys >= 0) & (ys <= height - 1)
    if not inside.any():
        return None
    return xs[inside], ys[inside]


def _score(gray: np.ndarray, samples: List[Tuple[np.ndarray, np.ndarray]], dy: int) -> float:
    """Mean gray value under all staff lines of a stave when shifted by ``dy`` pixels.

    Dark ink is a low gray value, so the best fit is the *minimum*.
    """
    total, count = 0.0, 0
    height = gray.shape[0]
    for xs, ys in samples:
        rows = np.clip(np.rint(ys + dy).astype(int), 0, height - 1)
        total += float(gray[rows, xs].sum())
        count += len(xs)
    return total / count if count else float('inf')


def compute_shift(gray: np.ndarray,
                  samples: List[Tuple[np.ndarray, np.ndarray]],
                  max_shift: int) -> Tuple[int, float, float]:
    """Find the vertical shift (in pixels) that puts a stave onto the most ink.

    All staff lines of the stave are moved by the same amount, which is what keeps their
    relative geometry -- and with it the position of every symbol in the staff -- intact.
    Returns ``(dy, score_at_0, score_at_dy)``.
    """
    score_0 = _score(gray, samples, 0)
    if max_shift <= 0 or not samples:
        return 0, score_0, score_0

    candidates = np.arange(-max_shift, max_shift + 1)
    scores = np.array([_score(gray, samples, int(dy)) for dy in candidates])

    spread = float(scores.max() - scores.min())
    if spread < MIN_CONTRAST:
        return 0, score_0, score_0

    threshold = scores.min() + SCORE_TOLERANCE * spread
    within = candidates[scores <= threshold]
    # Smallest correction that is as good as the best one: staff line detection is already
    # accurate most of the time, so ties must not move a stave.
    dy = int(within[np.argmin(np.abs(within))])
    return dy, score_0, float(scores[candidates == dy][0])


def _apply_shift(line: Line, dy: float, move_symbols: bool):
    """Move a stave down by ``dy`` in page coordinates.

    Only the y of the existing polyline points changes -- no point is added, removed or
    reordered -- so the shape of every staff line survives the correction.
    """
    for staff_line in line.staff_lines:
        if len(staff_line.coords.points) > 0:
            staff_line.coords.points[:, 1] += dy
            staff_line.update()

    if len(line.coords.points) > 0:
        line.coords.points[:, 1] += dy

    symbols = list(line.symbols) + list(line.additional_symbols)
    if move_symbols:
        # Keep every symbol where it is relative to its stave: coordinates and
        # positionInStaff stay consistent and nothing the user annotated changes meaning.
        for symbol in symbols:
            symbol.coord = Point(symbol.coord.x, symbol.coord.y + dy)
    else:
        # Symbols keep their absolute position on the page, so their position in the staff
        # -- and thus their pitch -- is re-derived from the corrected staff lines.
        for symbol in symbols:
            symbol.position_in_staff = line.staff_lines.compute_position_in_staff(
                symbol.coord, clef=symbol.symbol_type == SymbolType.CLEF)

    line.aabb = line._compute_aabb()


@dataclass(frozen=True)
class StaffLineCorrectionResult(AlgorithmPredictionResult):
    page: DatabasePage
    shifts: List[LineShift]
    move_symbols: bool

    def to_dict(self):
        # Only the delta is reported: the client applies it to the staff lines it already
        # holds, which keeps all ids, symbols and text of the page untouched.
        return {
            'page': self.page.page,
            'book': self.page.book.book,
            'moveSymbols': self.move_symbols,
            'lines': [{'id': s.line_id, 'dy': s.dy} for s in self.shifts if s.dy != 0],
        }

    def store_to_page(self):
        pcgts = self.page.pcgts()
        by_id = {s.line_id: s.dy for s in self.shifts if s.dy != 0}
        if not by_id:
            return

        for block in pcgts.page.music_blocks():
            changed = False
            for line in block.lines:
                dy = by_id.get(line.id)
                if dy:
                    _apply_shift(line, dy, self.move_symbols)
                    changed = True
            if changed:
                block.aabb = block._compute_aabb()

        if not self.move_symbols:
            pcgts.page.update_note_names()

        pcgts.to_file(self.page.file('pcgts').local_path())


class StaffLineCorrectionPredictor(AlgorithmPredictor):
    """Snaps already detected staves onto the ink of the page.

    Staff line detection is usually accurate but now and then produces a stave that is
    correct in shape yet shifted up or down by a few pixels as a whole. This step searches a
    small window of vertical shifts per stave and keeps the one under which the staff lines
    cover the most ink. It never adds, removes or reshapes anything -- it only changes the y
    of the points a stave already has.
    """

    @staticmethod
    def meta() -> Meta.__class__:
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        progress = PredictionProgress(callback, len(pages))
        progress.start()

        for page in pages:
            try:
                yield StaffLineCorrectionResult(
                    page=page,
                    shifts=self._predict_page(page),
                    move_symbols=self.params.staffLineCorrectionMoveSymbols,
                )
            except Exception as e:
                # One unreadable image must not abort a whole book run.
                logger.exception("Staff line correction failed for page {}".format(page.page))
                yield FailedPageResult(page_name=page.page, book_name=page.book.book, error=str(e))
            progress.page_finished()

    def _predict_page(self, page: DatabasePage) -> List[LineShift]:
        from PIL import Image

        pcgts = page.pcgts()
        lines = pcgts.page.all_music_lines()
        if not lines:
            return []

        gray = np.array(Image.open(
            page.file(SCALE_REFERENCE.file('gray'), create_if_not_existing=True).local_path()).convert('L'),
            dtype=float)
        height, width = gray.shape[:2]

        max_shift_rel = min(max(self.params.staffLineCorrectionMaxShift, 0.0), 0.9)

        shifts = []
        for line in lines:
            samples = []
            for staff_line in line.staff_lines:
                sample = _line_samples(pcgts.page.page_to_image_scale(staff_line.coords, SCALE_REFERENCE),
                                       width, height)
                if sample is not None:
                    samples.append(sample)

            line_distance = pcgts.page.page_to_image_scale(line.avg_line_distance(default=0), SCALE_REFERENCE)
            # A shift of a full staff space would fit every staff line onto its neighbour and
            # score perfectly while being plainly wrong, hence the cap well below 1.0 above.
            max_shift = int(round(max_shift_rel * line_distance))
            if not samples or max_shift <= 0:
                continue

            dy_px, before, after = compute_shift(gray, samples, max_shift)
            if dy_px == 0:
                continue

            shifts.append(LineShift(
                line_id=line.id,
                dy=pcgts.page.image_to_page_scale(float(dy_px), SCALE_REFERENCE),
                dy_px=dy_px,
                score_before=before,
                score_after=after,
            ))

        logger.debug("Staff line correction shifted {} of {} staves of page {}".format(
            len(shifts), len(lines), page.page))
        return shifts

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        # Anything with staves can be corrected; there is no state marking a page as
        # already corrected, so "unprocessed" means "has something to correct".
        return len(page.pcgts().page.music_blocks()) > 0


if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    from database import DatabaseBook

    book = DatabaseBook('demo')
    predictor = StaffLineCorrectionPredictor(AlgorithmPredictorSettings(Meta.best_model_for_book(book)))
    for result in predictor.predict(book.pages()[:1]):
        print(result.to_dict())
