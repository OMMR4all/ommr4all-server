"""Page level text transcription using LLMs.

Unlike the line based OCR predictors (calamari/nautilus/guppy) this predictor
sends the complete page image to a (vision) LLM and afterwards assigns the
returned text lines to the existing PCGTS text line regions:

* if the LLM returned bounding boxes, each LLM line is assigned to the text
  line region with the best geometric overlap,
* otherwise the LLM lines are matched to the regions in reading order.
"""
import logging
import math
import os

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

from PIL import Image

from database import DatabasePage
from database.file_formats.pcgts import PcGts, PageScaleReference
from database.file_formats.pcgts.page import Sentence
from database.file_formats.pcgts.page.block import BlockType
from database.file_formats.pcgts.page.line import Line
from database.file_formats.pcgts.page.page import Page
from omr.steps.algorithm import AlgorithmMeta, AlgorithmPredictionResult, AlgorithmPredictionResultGenerator, \
    AlgorithmPredictor, FailedPageResult, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.correction_tools.dictionary_corrector.predictor import DictionaryCorrector
from omr.steps.text.hyphenation.hyphenator import CombinedHyphenator, HyphenDicts
from omr.steps.text.llm.adapters import LLMPageTranscription, LLMTextLine, create_adapter

logger = logging.getLogger(__name__)

# block types whose lines are filled by the LLM transcription
LLM_TEXT_BLOCK_TYPES = [BlockType.LYRICS, BlockType.PARAGRAPH, BlockType.HEADING]


class SingleLineLLMResult(NamedTuple):
    line: Line
    hyphenated: str
    raw_text: str

    def to_dict(self):
        return {'sentence': self.hyphenated,
                'id': self.line.id,
                }


@dataclass(frozen=True)
class PredictionResult(AlgorithmPredictionResult):
    pcgts: PcGts
    dataset_page: DatabasePage
    text_lines: List[SingleLineLLMResult]

    def to_dict(self):
        return {'textLines': [l.to_dict() for l in self.text_lines]}

    def store_to_page(self):
        for line in self.text_lines:
            line.line.sentence = Sentence.from_string(line.hyphenated)
        self.pcgts.page.annotations.connections.clear()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


def sort_lines_in_reading_order(lines: List[Line]) -> List[Line]:
    """Group the lines into columns by horizontal overlap and return them
    column by column (left to right), top to bottom within a column."""
    columns = []
    for line in lines:
        left, right = line.aabb.left(), line.aabb.right()
        matching = [c for c in columns if c['left'] <= right and c['right'] >= left]
        columns.append({'lines': sum([c['lines'] for c in matching], [line]),
                        'left': min([left] + [c['left'] for c in matching]),
                        'right': max([right] + [c['right'] for c in matching]),
                        })
        for c in matching:
            columns.remove(c)

    columns.sort(key=lambda c: c['left'])
    ordered = []
    for c in columns:
        ordered += sorted(c['lines'], key=lambda l: l.aabb.top())
    return ordered


def assign_llm_lines_to_regions(page: Page,
                                target_lines: List[Line],
                                llm_lines: List[LLMTextLine],
                                scale_reference: PageScaleReference) -> Dict[str, str]:
    """Returns a mapping line.id -> transcribed text."""
    target_lines = sort_lines_in_reading_order(target_lines)
    with_bbox = [l for l in llm_lines if l.bbox is not None]

    logger.info("LLM alignment: %d LLM lines (%d with bbox) -> %d target line regions",
                len(llm_lines), len(with_bbox), len(target_lines))
    for i, line in enumerate(target_lines):
        aabb = line.aabb
        logger.info("  target[%d] id=%s aabb=(l=%.4f t=%.4f r=%.4f b=%.4f)",
                    i, line.id, aabb.left(), aabb.top(), aabb.right(), aabb.bottom())

    if with_bbox and len(with_bbox) >= len(llm_lines) * 0.5:
        logger.info("LLM alignment strategy: bbox overlap (%d/%d lines have a bbox)",
                    len(with_bbox), len(llm_lines))
        assignment = _assign_by_bbox(page, target_lines, with_bbox, scale_reference)
    else:
        logger.info("LLM alignment strategy: reading order (only %d/%d lines have a bbox)",
                    len(with_bbox), len(llm_lines))
        assignment = _assign_by_order(target_lines, llm_lines)

    for line_id, text in assignment.items():
        logger.info("LLM alignment result: line %s -> %r", line_id, text)
    unassigned = [l.id for l in target_lines if l.id not in assignment]
    if unassigned:
        logger.info("LLM alignment: %d target lines got no text: %s", len(unassigned), unassigned)
    return assignment


def _split_words_between_lines(text: str,
                               x1: float, x2: float,
                               candidates: List[Line]) -> List[Tuple[Line, float, str]]:
    """Distribute the words of one LLM text line over several side-by-side
    target lines: each word's x-position is estimated by its character offset
    within the llm bbox and the word goes to the line whose x-range is
    closest. Returns (line, x-position, text) per line that received words."""
    words = text.split()
    total_chars = sum(len(w) for w in words) + max(0, len(words) - 1)
    per_line: Dict[str, List[Tuple[float, str]]] = {}
    by_id = {l.id: l for l in candidates}
    pos = 0
    for word in words:
        wx = x1 + ((pos + len(word) / 2) / total_chars) * (x2 - x1)

        def x_dist(line: Line) -> float:
            aabb = line.aabb
            if aabb.left() <= wx <= aabb.right():
                return 0.0
            return min(abs(wx - aabb.left()), abs(wx - aabb.right()))

        target = min(candidates, key=x_dist)
        per_line.setdefault(target.id, []).append((wx, word))
        pos += len(word) + 1
    return [(by_id[lid], parts[0][0], ' '.join(w for _, w in parts))
            for lid, parts in per_line.items()]


def _assign_by_bbox(page: Page,
                    target_lines: List[Line],
                    llm_lines: List[LLMTextLine],
                    scale_reference: PageScaleReference) -> Dict[str, str]:
    def to_page(v: float) -> float:
        return page.image_to_page_scale(v, scale_reference)

    # collect per region: list of (x-position, text) to keep the order within a line
    collected: Dict[str, List[Tuple[float, str]]] = {}

    for llm_line in llm_lines:
        x1, y1, x2, y2 = [to_page(v) for v in llm_line.bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        best, best_score = None, 0.0
        strong: List[Line] = []  # lines mostly covered by the llm box (same text row)
        for line in target_lines:
            aabb = line.aabb
            # intersection over area of the llm box
            ix = max(0.0, min(x2, aabb.right()) - max(x1, aabb.left()))
            iy = max(0.0, min(y2, aabb.bottom()) - max(y1, aabb.top()))
            area = (x2 - x1) * (y2 - y1)
            score = (ix * iy) / area if area > 0 else 0.0
            if score > best_score:
                best, best_score = line, score
            w = aabb.right() - aabb.left()
            h = aabb.bottom() - aabb.top()
            if w > 0 and h > 0 and ix >= 0.5 * w and iy >= 0.5 * min(h, y2 - y1):
                strong.append(line)

        # one llm line spanning several target lines that sit side by side on
        # the same y-level (e.g. lyrics split at a drop capital): distribute
        # the words by their estimated x-position instead of winner-takes-all
        if len(strong) >= 2:
            strong.sort(key=lambda l: l.aabb.left())
            x_disjoint = all(nxt.aabb.left() >= prv.aabb.right()
                             - 0.2 * min(prv.aabb.right() - prv.aabb.left(),
                                         nxt.aabb.right() - nxt.aabb.left())
                             for prv, nxt in zip(strong, strong[1:]))
            if x_disjoint:
                for line, wx, part in _split_words_between_lines(llm_line.text, x1, x2, strong):
                    logger.info("LLM bbox assign (split %d-way): %r -> line %s", len(strong), part, line.id)
                    collected.setdefault(line.id, []).append((wx, part))
                continue

        if best is None:
            # no overlap at all: fall back to the closest line center
            def dist(line: Line) -> float:
                aabb = line.aabb
                lx, ly = (aabb.left() + aabb.right()) / 2, (aabb.top() + aabb.bottom()) / 2
                return (lx - cx) ** 2 + (ly - cy) ** 2

            best = min(target_lines, key=dist, default=None)
            if best is None:
                continue
            logger.warning("LLM text line '%s' has no overlap with any text region, assigned to closest line %s",
                           llm_line.text, best.id)
        else:
            logger.info("LLM bbox assign: %r bbox_px=%s bbox_page=(%.4f, %.4f, %.4f, %.4f) -> line %s (overlap=%.2f)",
                        llm_line.text, llm_line.bbox, x1, y1, x2, y2, best.id, best_score)

        collected.setdefault(best.id, []).append((x1, llm_line.text))

    return {line_id: ' '.join(t for _, t in sorted(parts, key=lambda p: p[0]))
            for line_id, parts in collected.items()}


def join_spaced_syllables(text: str, word_dictionary: Optional[Dict[str, str]]) -> str:
    """In manuscripts the syllables of a word are often written spaced apart
    (aligned under the neumes) and the LLM transcribes them as separate
    tokens, e.g. 'Al le lu ia'. Greedily merge runs of consecutive tokens
    whose concatenation is a known word of the lyrics dictionary (longest
    match first), so hyphenation afterwards yields 'Al-le-lu-ia'."""
    if not word_dictionary:
        return text
    tokens = text.split()
    out: List[str] = []
    i = 0
    while i < len(tokens):
        merged = None
        for j in range(min(len(tokens), i + 8), i + 1, -1):
            candidate = ''.join(tokens[i:j])
            key = candidate.lower().strip('.,;:!?')
            if len(key) >= 3 and key.isalpha() and key in word_dictionary:
                merged = candidate
                break
        if merged is not None:
            logger.info("LLM syllable join: %r -> %r", ' '.join(tokens[i:j]), merged)
            out.append(merged)
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)


def _assign_by_order(target_lines: List[Line], llm_lines: List[LLMTextLine]) -> Dict[str, str]:
    llm_texts = [l.text for l in llm_lines if l.text.strip()]
    if len(llm_texts) != len(target_lines):
        logger.warning("LLM returned %d text lines but the page has %d text line regions. "
                       "Assigning in reading order, remaining entries are dropped.",
                       len(llm_texts), len(target_lines))
    for line, text in zip(target_lines, llm_texts):
        logger.info("LLM order assign: line %s <- %r", line.id, text)
    for text in llm_texts[len(target_lines):]:
        logger.info("LLM order assign: dropped (no region left): %r", text)
    return {line.id: text for line, text in zip(target_lines, llm_texts)}


def limit_pixels(image: Image.Image, max_pixels: int) -> Tuple[Image.Image, float]:
    """Downscale ``image`` to at most ``max_pixels``, preserving the aspect ratio.

    Returns the image and the factor it was scaled by, so coordinates reported
    against the returned image can be divided back into the original space.
    """
    w, h = image.size
    pixels = w * h
    if max_pixels <= 0 or pixels <= max_pixels:
        return image, 1.0

    scale = math.sqrt(max_pixels / pixels)
    size = (max(int(w * scale), 1), max(int(h * scale), 1))
    logger.info('Scaling the page image down from %dx%d to %dx%d to stay within %d pixels',
                w, h, size[0], size[1], max_pixels)
    resized = image.resize(size, Image.BILINEAR)
    # the achieved factor, not the requested one: int() truncates
    return resized, resized.size[0] / w


def rescale_line(line: LLMTextLine, factor: float) -> LLMTextLine:
    """The same line with its bbox scaled, e.g. back from a downscaled image."""
    if line.bbox is None or factor == 1.0:
        return line
    return LLMTextLine(text=line.text, bbox=tuple(v * factor for v in line.bbox))


def is_out_of_memory(e: BaseException) -> bool:
    """Whether an exception is a GPU/host out-of-memory condition.

    Matched by name and message so that the module does not have to import torch,
    which the API based adapters do not need at all.
    """
    return type(e).__name__ in ('OutOfMemoryError', 'CudaOutOfMemoryError') \
        or 'out of memory' in str(e).lower()


def free_gpu_memory() -> None:
    """Best effort release of cached GPU blocks between attempts."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class LLMTextPredictor(AlgorithmPredictor):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from omr.steps.text.llm.meta import Meta
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        # api keys and endpoint urls are intentionally not taken from the request:
        # they are configured on the server via environment variables only
        # (GEMINI_API_KEY, OPENAI_API_KEY, OPENAI_API_URL, ...)
        self.adapter = create_adapter(
            provider=self.params.llmProvider,
            model=self.params.llmModel,
            prompt=self.params.llmCustomPrompt,
        )
        # The normalised image, not the raw high resolution scan: it is scaled to a fixed
        # number of pixels per staff line distance, so the text has the same pixel height
        # in every book, and a page stays small enough for a vision tower that attends
        # globally over its patches (a full 2500 px page needs tens of GiB for that).
        self.scale_reference = PageScaleReference.NORMALIZED_X2
        self.dict_corrector = None

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return any(len(l.sentence.syllables) == 0
                   for l in page.pcgts().page.all_lines_by_type(LLM_TEXT_BLOCK_TYPES))

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1, right=1)

        for i, db_page in enumerate(pages):
            try:
                yield self._predict_page(db_page, hyphen)
            except Exception as e:
                # one unreadable page must not abort the run: report it and carry on.
                # TaskRunnerPrediction filters these out and passes them to the client
                # as skipped pages.
                logger.exception(e)
                yield FailedPageResult(db_page.page, db_page.book.book,
                                       '{}: {}'.format(type(e).__name__, e))

            if callback:
                callback.progress_updated((i + 1) / len(pages), n_pages=len(pages), n_processed_pages=i + 1)

    def _predict_page(self, db_page: DatabasePage, hyphen: CombinedHyphenator) -> 'PredictionResult':
        pcgts = db_page.pcgts()
        page = pcgts.page

        if self.settings.params.useDictionaryCorrection and self.dict_corrector is None:
            self.dict_corrector = DictionaryCorrector(hyphenator=hyphen)
            self.dict_corrector.load_dict(book=db_page.book)

        target_lines = page.all_lines_by_type(LLM_TEXT_BLOCK_TYPES)
        text_lines: List[SingleLineLLMResult] = []

        if len(target_lines) > 0:
            image_path = db_page.file(self.scale_reference.file('color'), create_if_not_existing=True).local_path()
            image = Image.open(image_path).convert('RGB')

            transcription, scale = self._transcribe(image, db_page)
            logger.info("LLM raw response for page %s (image %dx%d px):\n%s",
                        db_page.page, image.width, image.height, transcription.raw_response)
            logger.info("LLM transcription of page %s parsed into %d lines:",
                        db_page.page, len(transcription.lines))
            lines = transcription.lines
            if scale != 1.0:
                # the adapter reported boxes against the downscaled image it was given
                lines = [rescale_line(l, 1.0 / scale) for l in lines]
            for n, llm_line in enumerate(lines):
                logger.info("  llm[%d] bbox=%s text=%r", n, llm_line.bbox, llm_line.text)

            assignment = assign_llm_lines_to_regions(page, target_lines, lines,
                                                     self.scale_reference)

            for line in target_lines:
                text = assignment.get(line.id, '')
                if not text:
                    continue
                text = join_spaced_syllables(text, hyphen.dictionary)
                if self.dict_corrector:
                    hyphenated = self.dict_corrector.segmentate_correct_and_hyphenate_text(text)
                else:
                    hyphenated = hyphen.apply_to_sentence(text)
                text_lines.append(SingleLineLLMResult(line=line, hyphenated=hyphenated, raw_text=text))

        return PredictionResult(pcgts=pcgts, dataset_page=db_page, text_lines=text_lines)

    def _transcribe(self, image: Image.Image, db_page: DatabasePage) -> Tuple[LLMPageTranscription, float]:
        """Transcribe the page, halving the pixel budget once if the model runs out of memory.

        color_norm_x2 has no size cap of its own -- it is the high resolution page divided
        by the measured staff line distance -- so a page whose normalisation went wrong can
        still be far too large for the vision model.
        """
        from ommr4all.settings import LLM_MAX_IMAGE_PIXELS

        budget = LLM_MAX_IMAGE_PIXELS
        for attempt in range(2):
            scaled, scale = limit_pixels(image, budget)
            try:
                return self.adapter.transcribe(scaled), scale
            except Exception as e:
                if attempt == 0 and is_out_of_memory(e):
                    budget = max(scaled.size[0] * scaled.size[1] // 2, 1)
                    logger.warning('Out of memory transcribing page %s at %dx%d; retrying with at '
                                   'most %d pixels. Lower OMMR4ALL_LLM_MAX_PIXELS to avoid this.',
                                   db_page.page, scaled.size[0], scaled.size[1], budget)
                    free_gpu_memory()
                    continue
                raise


if __name__ == '__main__':
    from database import DatabaseBook
    from omr.steps.step import Step, AlgorithmTypes
    from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams

    book = DatabaseBook('demo')
    meta = Step.meta(AlgorithmTypes.OCR_LLM)
    settings = AlgorithmPredictorSettings(model=meta.best_model_for_book(book))
    settings.params.llmProvider = 'chandra'
    pred = LLMTextPredictor(settings)
    for result in pred.predict(book.pages()[0:1]):
        for tl in result.text_lines:
            print(tl.line.id, tl.hyphenated)
