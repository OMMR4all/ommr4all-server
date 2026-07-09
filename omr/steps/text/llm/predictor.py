"""Page level text transcription using LLMs.

Unlike the line based OCR predictors (calamari/nautilus/guppy) this predictor
sends the complete page image to a (vision) LLM and afterwards assigns the
returned text lines to the existing PCGTS text line regions:

* if the LLM returned bounding boxes, each LLM line is assigned to the text
  line region with the best geometric overlap,
* otherwise the LLM lines are matched to the regions in reading order.
"""
import logging
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
    AlgorithmPredictor, PredictionCallback
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.text.correction_tools.dictionary_corrector.predictor import DictionaryCorrector
from omr.steps.text.hyphenation.hyphenator import CombinedHyphenator, HyphenDicts
from omr.steps.text.llm.adapters import LLMTextLine, create_adapter

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

    if with_bbox and len(with_bbox) >= len(llm_lines) * 0.5:
        return _assign_by_bbox(page, target_lines, with_bbox, scale_reference)
    return _assign_by_order(target_lines, llm_lines)


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
        for line in target_lines:
            aabb = line.aabb
            # intersection over area of the llm box
            ix = max(0.0, min(x2, aabb.right()) - max(x1, aabb.left()))
            iy = max(0.0, min(y2, aabb.bottom()) - max(y1, aabb.top()))
            area = (x2 - x1) * (y2 - y1)
            score = (ix * iy) / area if area > 0 else 0.0
            if score > best_score:
                best, best_score = line, score

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

        collected.setdefault(best.id, []).append((x1, llm_line.text))

    return {line_id: ' '.join(t for _, t in sorted(parts, key=lambda p: p[0]))
            for line_id, parts in collected.items()}


def _assign_by_order(target_lines: List[Line], llm_lines: List[LLMTextLine]) -> Dict[str, str]:
    llm_texts = [l.text for l in llm_lines if l.text.strip()]
    if len(llm_texts) != len(target_lines):
        logger.warning("LLM returned %d text lines but the page has %d text line regions. "
                       "Assigning in reading order, remaining entries are dropped.",
                       len(llm_texts), len(target_lines))
    return {line.id: text for line, text in zip(target_lines, llm_texts)}


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
        self.scale_reference = PageScaleReference.HIGHRES
        self.dict_corrector = None

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return any(len(l.sentence.syllables) == 0
                   for l in page.pcgts().page.all_lines_by_type(LLM_TEXT_BLOCK_TYPES))

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1, right=1)

        for i, db_page in enumerate(pages):
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

                transcription = self.adapter.transcribe(image)
                logger.debug("LLM transcription of page %s returned %d lines",
                             db_page.page, len(transcription.lines))

                assignment = assign_llm_lines_to_regions(page, target_lines, transcription.lines,
                                                         self.scale_reference)

                for line in target_lines:
                    text = assignment.get(line.id, '')
                    if not text:
                        continue
                    if self.dict_corrector:
                        hyphenated = self.dict_corrector.segmentate_correct_and_hyphenate_text(text)
                    else:
                        hyphenated = hyphen.apply_to_sentence(text)
                    text_lines.append(SingleLineLLMResult(line=line, hyphenated=hyphenated, raw_text=text))

            if callback:
                callback.progress_updated((i + 1) / len(pages), n_pages=len(pages), n_processed_pages=i + 1)

            yield PredictionResult(pcgts=pcgts, dataset_page=db_page, text_lines=text_lines)


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
