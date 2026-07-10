import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from PIL import Image

from database import DatabasePage
from database.file_formats import PcGts
from database.file_formats.pcgts import BlockType
from database.file_formats.pcgts.page import Line, MusicSymbol, Sentence, Syllable, SyllableConnection, SymbolType
from database.file_formats.pcgts.page.annotations import Annotations, SyllableConnector
from database.file_formats.performance.pageprogress import Locks
from omr.end2end.codec.parse import symbol_parse
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    AlgorithmPredictionResultGenerator, PredictionCallback

from .sample_builder import find_text_region_for_music_block, crop_block_image

logger = logging.getLogger(__name__)


class Chunk(NamedTuple):
    text: str  # syllable text; '' for the leading symbols-only chunk
    connection: SyllableConnection
    symbols: List[MusicSymbol]


CHUNK_RE = re.compile(r"(?P<sep>[ \-]?)(?P<text>[^\[\]]*)\[(?P<syms>[^\[\]]*)\]")
TUPLE_RE = re.compile(r"\(([^()]+)\)")


def parse_chunks(decoded: str, skip_gapped: bool = True) -> List[Chunk]:
    """Parses a decoded codec string like ``*[(clef|f|7)] quos[(note|0|6|2)]-ci[(note|0|9|2)]``
    into syllable chunks with their music symbols. Separator ' ' starts a new syllable,
    '-' continues the previous one (mirrors the GT builder in data_generator.py)."""
    content = decoded.replace("[NEW_CHANT]", " ").replace("[NewLine]", " ").replace("NEW_CHANT", " ")

    chunks = []
    for m in CHUNK_RE.finditer(content):
        text = m.group("text").strip()
        symbols = []
        for tuple_str in TUPLE_RE.findall(m.group("syms")):
            s = symbol_parse(tuple_str, skip_gapped=skip_gapped)
            if s:
                symbols.append(s)

        if text in ("*", ""):
            # leading music block before the first syllable anchor
            chunks.append(Chunk(text="", connection=SyllableConnection.NEW, symbols=symbols))
            continue

        connection = SyllableConnection.HIDDEN if m.group("sep") == "-" else SyllableConnection.NEW
        chunks.append(Chunk(text=text, connection=connection, symbols=symbols))
    return chunks


def assign_symbol_coords(chunks: List[Chunk], music_line: Line) -> List[MusicSymbol]:
    """Synthesizes coordinates for decoded symbols: each chunk gets an equal horizontal slot
    along the staff, symbols are evenly spaced inside their slot, and y is derived from the
    predicted position in staff."""
    staff_lines = music_line.staff_lines
    symbols = [s for c in chunks for s in c.symbols]
    if len(staff_lines) == 0 or len(symbols) == 0:
        return symbols

    aabb = staff_lines.aabb()
    left, right = aabb.left(), aabb.right()
    populated = [c for c in chunks if len(c.symbols) > 0]
    slot_w = (right - left) / max(len(populated), 1)

    for ci, chunk in enumerate(populated):
        slot_left = left + ci * slot_w
        n = len(chunk.symbols)
        for si, s in enumerate(chunk.symbols):
            x = slot_left + (si + 1) / (n + 1) * slot_w
            s.coord = staff_lines.compute_coord_by_position_in_staff(x, s.position_in_staff)
    return symbols


class SingleBlockResult(NamedTuple):
    music_line: Line
    text_line: Optional[Line]
    symbols: List[MusicSymbol]
    sentence: Optional[Sentence]
    chunks: List[Chunk]


@dataclass(frozen=True)
class PredictionResult(AlgorithmPredictionResult):
    pcgts: PcGts
    dataset_page: DatabasePage
    blocks: List[SingleBlockResult]
    annotations: Annotations

    def to_dict(self):
        return {
            'musicLines': [{'id': b.music_line.id,
                            'symbols': [s.to_json() for s in b.symbols]} for b in self.blocks],
            # full syllable JSON (ids included) so the client can resolve the annotation connectors
            'textLines': [{'id': b.text_line.id,
                           'sentence': b.sentence.text(),
                           'syllables': b.sentence.to_json()} for b in self.blocks
                          if b.text_line is not None and b.sentence is not None],
            'annotations': self.annotations.to_json(),
        }

    def store_to_page(self):
        page = self.pcgts.page
        for b in self.blocks:
            b.music_line.symbols = b.symbols
            if b.text_line is not None and b.sentence is not None:
                b.text_line.sentence = b.sentence
        page.annotations.connections.clear()
        page.annotations.connections.extend(self.annotations.connections)
        page.update_note_names()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class End2EndPredictor(AlgorithmPredictor, ABC):
    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return all(len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines())

    @classmethod
    def unlocked(cls, page: DatabasePage) -> bool:
        # writes symbols, text and syllables, so both locks must be open
        locked = page.page_progress().locked
        return not locked.get(Locks.SYMBOLS) and not locked.get(Locks.TEXT)

    @abstractmethod
    def _predict_crop(self, image: Image.Image) -> str:
        """Runs the network on a combined music+lyric crop and returns the decoded codec string."""
        pass

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        for page_i, db_page in enumerate(pages):
            pcgts = db_page.pcgts()
            page = pcgts.page
            annotations = Annotations(page)
            block_results = []

            try:
                full_image = Image.open(db_page.file("color_original").local_path())
            except Exception as e:
                logger.warning(f"Skipping page {db_page.page}: could not open image ({e})")
                yield PredictionResult(pcgts=pcgts, dataset_page=db_page, blocks=[], annotations=annotations)
                continue

            music_blocks = [b for b in page.blocks_of_type([BlockType.MUSIC]) if len(b.lines) > 0]
            for block_i, block in enumerate(music_blocks):
                music_line = block.lines[0]
                text_region = find_text_region_for_music_block(page, block)
                image = crop_block_image(full_image, page, block, text_region)

                decoded = self._predict_crop(image)
                chunks = parse_chunks(decoded)
                symbols = assign_symbol_coords(chunks, music_line)

                text_line = text_region.lines[0] if text_region is not None and len(text_region.lines) > 0 else None
                sentence = None
                if text_line is not None:
                    syllable_chunks = [c for c in chunks if c.text]
                    sentence = Sentence([Syllable(text=c.text, connection=c.connection) for c in syllable_chunks])
                    connection = annotations.get_or_create_connection(block, text_region)
                    for chunk, syllable in zip(syllable_chunks, sentence.syllables):
                        note = next((s for s in chunk.symbols if s.symbol_type == SymbolType.NOTE), None)
                        if note is not None:
                            connection.syllable_connections.append(SyllableConnector(syllable, note))

                block_results.append(SingleBlockResult(music_line=music_line, text_line=text_line,
                                                       symbols=symbols, sentence=sentence, chunks=chunks))
                if callback:
                    callback.progress_updated((page_i + (block_i + 1) / len(music_blocks)) / len(pages),
                                              n_pages=len(pages), n_processed_pages=page_i)

            yield PredictionResult(pcgts=pcgts, dataset_page=db_page, blocks=block_results, annotations=annotations)
