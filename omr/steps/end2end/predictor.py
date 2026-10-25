import logging
import math
import os
import re
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from database import DatabasePage
from database.file_formats import PcGts
from database.file_formats.pcgts import BlockType, LineStartType, MusicSymbolPositionInStaff, Point
from database.file_formats.pcgts.page import Block, Line, MusicSymbol, Page, Sentence, Syllable, SyllableConnection, \
    SymbolType
from database.file_formats.pcgts.page.coords import Coords
from database.file_formats.pcgts.page.annotations import Annotations, SyllableConnector
from database.file_formats.performance.pageprogress import Locks
from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.parse import symbol_parse
from omr.steps.algorithm import AlgorithmPredictor, AlgorithmPredictorSettings, AlgorithmPredictionResult, \
    AlgorithmPredictionResultGenerator, PredictionCallback, PredictionProgress

from .dataset import registered_class_specs
from .model_io import has_loc_feedback, has_localization, symbol_token_ids
from .sample_builder import find_text_region_for_music_block, compute_crop_box, crop_fraction_to_page_point, CropBox, \
    lyric_lines_for_music_block
from .torch_dataset import preprocess_pil_image

logger = logging.getLogger(__name__)

MAX_DECODE_LEN = 1500
# crops decoded together; sorted by width so that little of a batch is padding
PREDICT_BATCH_SIZE = 16
# hypotheses kept per crop; 1 = greedy decoding. 4 beams: +1 hSAR and better lyrics than greedy on
# Pa at ~20 % more prediction time; 8 beams add only noise-level gains (docs/end2end-experiments.md)
BEAM_SIZE = 4
# soft-argmax window around the attention peak in cells (`PointerLocalizationHead.locate`)
LOC_RADIUS = 1

# (x, y) of a symbol as fraction of the crop width and height
CropLoc = Tuple[float, float]


class Chunk(NamedTuple):
    text: str  # syllable text; '' for the leading symbols-only chunk
    connection: SyllableConnection
    symbols: List[MusicSymbol]
    locs: List[Optional[CropLoc]]  # per symbol, None if the model does not localise
    start: LineStartType = LineStartType.NONE  # a chant or strophe starts at this syllable


class Decoded(NamedTuple):
    text: str
    # character offset of a symbol token in `text` -> its predicted location
    locs: Dict[int, CropLoc]


CHUNK_RE = re.compile(r"(?P<sep>[ \-]?)(?P<text>[^\[\]]*)\[(?P<syms>[^\[\]]*)\]")
TUPLE_RE = re.compile(r"\(([^()]+)\)")
# markers without symbol content; blanked with equal length so character offsets stay valid
MARKER_RE = re.compile(r"\[NEW_CHANT\]|\[NEW_STROPHE\]|\[NewLine\]|NEW_CHANT")
MARKER_STARTS = {"[NEW_CHANT]": LineStartType.CHANT, "NEW_CHANT": LineStartType.CHANT,
                 "[NEW_STROPHE]": LineStartType.STROPHE}


def parse_chunks(decoded: str, skip_gapped: bool = True, class_specs: Optional[Dict] = None,
                 locs: Optional[Dict[int, CropLoc]] = None) -> List[Chunk]:
    """Parses a decoded codec string like ``*[(clef|f|7)] quos[(note|0|6|2)]-ci[(note|0|9|2)]``
    into syllable chunks with their music symbols. Separator ' ' starts a new syllable,
    '-' continues the previous one (mirrors the GT builder in data_generator.py). A chant or
    strophe marker sets `start` of the syllable chunk that follows it.

    `locs` maps the character offset of a symbol token to its location."""
    markers = [(m.start(), MARKER_STARTS.get(m.group())) for m in MARKER_RE.finditer(decoded)]
    markers = [(position, start) for position, start in markers if start is not None]
    content = MARKER_RE.sub(lambda m: " " * len(m.group()), decoded)
    locs = locs or {}

    chunks = []
    pending = LineStartType.NONE
    for m in CHUNK_RE.finditer(content):
        while markers and markers[0][0] < m.start("syms"):
            pending = markers.pop(0)[1]
        text = m.group("text").strip()
        symbols, symbol_locs = [], []
        for t in TUPLE_RE.finditer(m.group("syms")):
            s = symbol_parse(t.group(1), skip_gapped=skip_gapped, class_specs=class_specs)
            if s:
                symbols.append(s)
                symbol_locs.append(locs.get(m.start("syms") + t.start()))

        if text in ("*", ""):
            # leading music block before the first syllable anchor
            chunks.append(Chunk(text="", connection=SyllableConnection.NEW, symbols=symbols, locs=symbol_locs))
            continue

        connection = SyllableConnection.HIDDEN if m.group("sep") == "-" else SyllableConnection.NEW
        chunks.append(Chunk(text=text, connection=connection, symbols=symbols, locs=symbol_locs, start=pending))
        pending = LineStartType.NONE
    return chunks


def has_staff_position(s: MusicSymbol) -> bool:
    return MusicSymbolPositionInStaff.SPACE_0 <= s.position_in_staff <= MusicSymbolPositionInStaff.SPACE_7


def assign_symbol_coords(chunks: List[Chunk], music_line: Line, page: Page, box: CropBox) -> List[MusicSymbol]:
    """Places the decoded symbols on the page.

    A localised symbol takes x from the network and y from its predicted position in staff
    on the staff lines, so that the drawn position always agrees with the pitch; a symbol
    without a position in staff keeps the predicted y. Without localisation (models trained
    before the head existed) each chunk gets an equal horizontal slot along the staff,
    symbols are evenly spaced inside their slot."""
    staff_lines = music_line.staff_lines
    symbols = [s for c in chunks for s in c.symbols]
    if len(symbols) == 0:
        return symbols

    if all(l is not None for c in chunks for l in c.locs):
        for c in chunks:
            for s, (fx, fy) in zip(c.symbols, c.locs):
                x, y = crop_fraction_to_page_point(page, box, fx, fy)
                if len(staff_lines) > 0 and has_staff_position(s):
                    s.coord = staff_lines.compute_coord_by_position_in_staff(x, s.position_in_staff)
                else:
                    s.coord = Point(x, y)
        return symbols

    if len(staff_lines) == 0:
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


def chunk_x(chunk: Chunk) -> Optional[float]:
    """Page x of the chunk's first note (first symbol if it has no note), placed by `assign_symbol_coords`."""
    anchor = next((s for s in chunk.symbols if s.symbol_type == SymbolType.NOTE), None) or \
        next(iter(chunk.symbols), None)
    return anchor.coord.x if anchor is not None else None


def assign_chunks_to_lines(chunks: List[Chunk], lines: List[Line], start_tolerance: float = 0.0) -> List[int]:
    """Index into `lines` (sorted left to right) for every syllable chunk (`chunk.text`): the line
    whose horizontal extent contains the chunk's first note, else the nearest one. Lyrics read
    left to right, so a chunk never returns to an earlier line than its predecessor; a chunk
    without symbols stays on its predecessor's line. A chunk that starts a chant or strophe
    moves on to the next line if its note lies at most `start_tolerance` before that line: the
    first note of a chant often sits above the drop capital, left of where its lyric line begins."""
    def distance(line: Line, x: float) -> float:
        return max(line.aabb.left() - x, x - line.aabb.right(), 0)

    indices, current = [], 0
    for c in chunks:
        if not c.text:
            continue
        x = chunk_x(c)
        if x is not None and len(lines) > 1:
            current = max(current, min(range(len(lines)), key=lambda i: distance(lines[i], x)))
            if c.start != LineStartType.NONE and current + 1 < len(lines) and \
                    x >= lines[current + 1].aabb.left() - start_tolerance:
                current += 1
        indices.append(current)
    return indices


class SingleBlockResult(NamedTuple):
    music_line: Line
    symbols: List[MusicSymbol]
    chunks: List[Chunk]


class LineSplit(NamedTuple):
    index: int  # first syllable chunk of the new line
    x: float  # page x of the cut
    start: LineStartType


# a cut keeps this share of the line's width away from either end
SPLIT_MARGIN = 0.01
# a syllable that starts a chant or strophe joins the next lyric line if its note lies at most this
# many music block heights before it (on Pa_14819_gt such notes sat 0.23-0.38 heights before it)
START_TOLERANCE = 0.5


def plan_line_starts(chunks: List[Chunk], left: float, right: float,
                     drop_capitals: List[Tuple[float, float]]) -> Tuple[Optional[LineStartType], List[LineSplit]]:
    """For the syllable chunks of one lyric line (left to right, symbols placed): the start type
    of the line itself (None: its first syllable carries no marker) and a cut, left to right, at
    every marked syllable after the first that has a drop capital (x-interval of a DROP_CAPITAL
    block beside the line) between it and the previous syllable. The cut lies at the drop
    capital's left edge, as in the layout step, clamped into the line's extent [left, right].
    Without a drop capital the marker alone does not split: on Pa_14819_gt none of 21 such
    splits met a chant or strophe start of the GT."""
    own = chunks[0].start if chunks and chunks[0].start != LineStartType.NONE else None
    margin = SPLIT_MARGIN * (right - left)
    splits: List[LineSplit] = []
    for i in range(1, len(chunks)):
        if chunks[i].start == LineStartType.NONE:
            continue
        before = [s.coord.x for c in chunks[:i] for s in c.symbols]
        after = [s.coord.x for c in chunks[i:] for s in c.symbols]
        gap_left, gap_right = (before[-1] if before else left), (after[0] if after else right)
        dc = next((dc for dc in sorted(drop_capitals) if dc[0] < gap_right and dc[1] > gap_left), None)
        if dc is None:
            continue
        x = min(max(dc[0], left + margin), right - margin)
        # mislocated symbols can put a cut left of the previous one: the syllables stay together
        if splits and x <= splits[-1].x:
            continue
        splits.append(LineSplit(i, x, chunks[i].start))
    return own, splits


def cut_coords(coords: Coords, x: float) -> Optional[Tuple[Coords, Coords]]:
    """The parts of `coords` left and right of `x`, None if the cut does not divide it in two."""
    parts = coords.split_polygon_by_x(x)
    if any(len(p) == 0 for p in parts):
        return None
    left, right = sorted(parts, key=lambda p: p[:, 0].mean())
    # shapely closes the ring with a copy of the first point
    return Coords(left[:-1]), Coords(right[:-1])


class LyricLineResult(NamedTuple):
    line: Line
    sentence: Sentence
    block: Optional[Block] = None
    start_type: Optional[LineStartType] = None  # None: unchanged
    coords: Optional[Coords] = None  # new coords of a line that was split
    split_from: Optional[str] = None  # id of the line a new line was split from; `line` carries its coords

    def to_json(self) -> dict:
        # full syllable JSON (ids included) so the client can resolve the annotation connectors
        d = {'id': self.line.id, 'sentence': self.sentence.text(), 'syllables': self.sentence.to_json()}
        if self.start_type is not None:
            d['startType'] = self.start_type.value
        if self.split_from is not None:
            d.update(blockId=self.block.id, splitFrom=self.split_from, coords=self.line.coords.to_json())
        elif self.coords is not None:
            d['coords'] = self.coords.to_json()
        return d


def lyric_line_results(region: Block, line: Line, entries: List[Tuple[Block, Chunk]],
                       annotations: Annotations, drop_capitals: List[Block]) -> List[LyricLineResult]:
    """Sentences of a lyric line and of the lines split from it where a chant or strophe starts
    mid-line at a drop capital (`plan_line_starts`; `drop_capitals`: the page's DROP_CAPITAL
    blocks); `entries` are its syllable chunks with their music blocks, in reading order.
    Connects every syllable to its first note in `annotations`. The page is left untouched:
    `apply_text_lines` writes the result."""
    beside = [(dc.aabb.left(), dc.aabb.right()) for dc in drop_capitals
              if dc.aabb.bottom() > line.aabb.top() and dc.aabb.top() < line.aabb.bottom()]
    start, splits = plan_line_starts([c for _, c in entries], line.aabb.left(), line.aabb.right(), beside)
    parts: List[Tuple[int, Coords, LineStartType]] = []  # (first chunk, coords, start type) of every new line
    remaining = line.coords
    for split in splits:
        cut = cut_coords(remaining, split.x)
        if cut is None:
            continue
        if parts:
            parts[-1] = (parts[-1][0], cut[0], parts[-1][2])
            remaining = cut[1]
        else:
            source_coords, remaining = cut
        parts.append((split.index, remaining, split.start))

    bounds = [0] + [p[0] for p in parts] + [len(entries)]
    results = []
    for k in range(len(bounds) - 1):
        part_entries = entries[bounds[k]:bounds[k + 1]]
        sentence = Sentence([Syllable(text=c.text, connection=c.connection) for _, c in part_entries])
        for (music_block, chunk), syllable in zip(part_entries, sentence.syllables):
            note = next((s for s in chunk.symbols if s.symbol_type == SymbolType.NOTE), None)
            if note is not None:
                annotations.get_or_create_connection(music_block, region).syllable_connections.append(
                    SyllableConnector(syllable, note))
        if k == 0:
            results.append(LyricLineResult(line, sentence, region, start, source_coords if parts else None))
        else:
            _, coords, part_start = parts[k - 1]
            results.append(LyricLineResult(Line(coords=coords, start_type=part_start), sentence, region, part_start,
                                           split_from=line.id))
    return results


def apply_text_lines(page: Page, text_lines: List[LyricLineResult]):
    """Writes lyric line results into `page`, resolving lines and blocks by id: coords and start
    types of existing lines, new lines after the line they were split from (in their block and
    in the reading order), and every sentence."""
    previous: Dict[str, Line] = {}  # source line id -> its rightmost part so far
    for t in text_lines:
        if t.split_from is None:
            line = page.text_line_by_id(t.line.id)
            previous[line.id] = line
            if t.coords is not None:
                line.coords = t.coords
                line.update_aabb()
        else:
            line = t.line
            after = previous[t.split_from]
            block = page.block_by_id(t.block.id)
            block.lines.insert(block.lines.index(after) + 1, line)
            block.update_aabb()
            reading_order = page.reading_order.reading_order
            if after in reading_order:
                reading_order.insert(reading_order.index(after) + 1, line)
            previous[t.split_from] = line
        if t.start_type is not None:
            line.start_type = t.start_type
        line.sentence = t.sentence


@dataclass(frozen=True)
class PredictionResult(AlgorithmPredictionResult):
    pcgts: PcGts
    dataset_page: DatabasePage
    blocks: List[SingleBlockResult]
    # every lyric line below a predicted music block with the syllables assigned to it (possibly
    # none), each followed by the lines split from it
    text_lines: List[LyricLineResult]
    annotations: Annotations

    def to_dict(self):
        return {
            'musicLines': [{'id': b.music_line.id,
                            'symbols': [s.to_json() for s in b.symbols]} for b in self.blocks],
            'textLines': [t.to_json() for t in self.text_lines],
            'annotations': self.annotations.to_json(),
        }

    def store_to_page(self):
        page = self.pcgts.page
        for b in self.blocks:
            b.music_line.symbols = b.symbols
        apply_text_lines(page, self.text_lines)
        page.annotations.connections.clear()
        page.annotations.connections.extend(self.annotations.connections)
        page.update_note_names()
        self.pcgts.to_file(self.dataset_page.file('pcgts').local_path())


class End2EndPredictor(AlgorithmPredictor, ABC):
    """Beam-search decoding (`BEAM_SIZE`) with any `End2EndNetwork`; the architecture comes from `meta().create_network`.
    The network localises symbols if its checkpoint carries the localisation head."""

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        if self.settings.model is None:
            raise FileNotFoundError("No end2end model available. Train a model for this book first.")

        model = self.settings.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = SmartTokenizer(model.local_file('codec.txt'))
        state = torch.load(model.local_file('model_best.pth'), map_location='cpu')
        self.model = self.meta().create_network(len(self.tokenizer), localization=has_localization(state),
                                                pretrained_backbone=False, loc_feedback=has_loc_feedback(state))
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # a model without dataset_params.json predates the trainer integration
        has_params = os.path.exists(model.local_file('dataset_params.json'))
        self.height = self.dataset_params.height if has_params and self.dataset_params.height > 0 \
            else self.meta().default_height()
        self.class_specs = {s.class_id: s for s in registered_class_specs(self.dataset_params.symbol_label_sets)}
        self.symbol_ids = symbol_token_ids(self.tokenizer)

    @classmethod
    def unprocessed(cls, page: DatabasePage) -> bool:
        return all(len(l.symbols) == 0 for l in page.pcgts().page.all_music_lines())

    @classmethod
    def unlocked(cls, page: DatabasePage) -> bool:
        # writes symbols, text and syllables, so both locks must be open
        locked = page.page_progress().locked
        return not locked.get(Locks.SYMBOLS) and not locked.get(Locks.TEXT)

    @classmethod
    def runnable(cls, page: DatabasePage) -> bool:
        # reads a music block together with its lyrics: needs staff lines and the lyric regions
        # of the layout step, otherwise it would only overwrite the page with music-only guesses
        p = page.pcgts().page
        return any(len(l.staff_lines) > 0 for l in p.all_music_lines()) and \
            len(p.blocks_of_type([BlockType.LYRICS])) > 0

    def _predict_crops(self, images: List[Image.Image]) -> List[Decoded]:
        """Runs the network on combined music+lyric crops: per crop the codec string and, if the
        model localises, the location of every symbol token. Batched over the crops."""
        order = sorted(range(len(images)), key=lambda i: images[i].size[0] / images[i].size[1])
        results: List[Optional[Decoded]] = [None] * len(images)
        for start in range(0, len(order), PREDICT_BATCH_SIZE):
            batch = order[start:start + PREDICT_BATCH_SIZE]
            for i, decoded in zip(batch, self._decode_batch([images[i] for i in batch])):
                results[i] = decoded
        return results

    def _decode_batch(self, images: List[Image.Image]) -> List[Decoded]:
        """Beam search over the crops of a batch; per crop the hypothesis with the highest total
        log probability wins. Every hypothesis carries its own symbol locations and, with location
        feedback, its own latest location. `BEAM_SIZE = 1` is plain greedy decoding."""
        model, tok, device, k = self.model, self.tokenizer, self.device, BEAM_SIZE
        tensors = [preprocess_pil_image(image, self.height) for image in images]
        max_w = max(t.shape[2] for t in tensors)
        imgs = torch.stack([F.pad(t, (0, max_w - t.shape[2])) for t in tensors]).to(device)
        n = len(images)
        rows = n * k  # hypothesis h of crop i is row i * k + h
        widths = torch.tensor([t.shape[2] for t in tensors], device=device)
        symbol_ids = torch.tensor(sorted(self.symbol_ids), device=device)
        amp = torch.autocast(device.type, dtype=torch.bfloat16,
                             enabled=device.type == 'cuda' and torch.cuda.is_bf16_supported())
        with torch.no_grad(), amp:
            enc = model.encode(imgs, widths)
            enc = enc._replace(**{f: getattr(enc, f).repeat_interleave(k, dim=0)
                                  for f in ('memory', 'memory_padding', 'loc_keys', 'loc_valid')
                                  if getattr(enc, f) is not None})
            if model.has_localization:
                centers = model.cell_centers(enc)
                w_cells = enc.loc_keys.shape[-1]
            cache = model.start_decoding(enc)
            tokens = torch.full((rows,), tok.SOS, dtype=torch.long, device=device)
            finished = torch.zeros(rows, dtype=torch.bool, device=device)
            # only the first hypothesis of a crop is alive at the start, the others would duplicate it
            scores = torch.full((n, k), float('-inf'), device=device)
            scores[:, 0] = 0
            # per hypothesis the emitted tokens and the location of each (NaN: none)
            history = torch.empty((rows, 0), dtype=torch.long, device=device)
            located_history = torch.empty((rows, 0, 2), device=device)
            # location of the latest symbol in units of the input height (NaN: none yet)
            previous = torch.full((rows, 2), float('nan'), device=device)
            # a finished hypothesis only continues with <pad> at no cost and so keeps its score
            finished_logp = torch.full((len(tok),), float('-inf'), device=device)
            finished_logp[tok.PAD] = 0
            for _ in range(MAX_DECODE_LEN):
                last = model.decode_next(tokens, cache, enc, previous)
                # the state whose input is a symbol token localises that symbol, the last emitted one
                localise = torch.isin(tokens, symbol_ids) & ~finished if model.has_localization else None
                if localise is not None and bool(localise.any()):
                    located = model.loc_head.locate(model.localization_logits(last, enc)[:, 0], centers, w_cells,
                                                    LOC_RADIUS)
                    previous = torch.where(localise[:, None], located, previous)
                    located_history[:, -1] = torch.where(localise[:, None], located, located_history[:, -1])
                logp = torch.log_softmax(model.logits(last)[:, 0].float(), dim=-1)
                logp = torch.where(finished[:, None], finished_logp, logp)
                vocab = logp.shape[-1]
                scores, top = (scores.view(rows, 1) + logp).view(n, k * vocab).topk(k, dim=-1)
                parents = (torch.arange(n, device=device)[:, None] * k + top // vocab).flatten()
                tokens = (top % vocab).flatten()
                finished = finished[parents] | (tokens == tok.EOS)
                previous = previous[parents]
                history = torch.cat([history[parents], tokens[:, None]], dim=1)
                located_history = torch.cat([located_history[parents],
                                             torch.full((rows, 1, 2), float('nan'), device=device)], dim=1)
                for c in cache:
                    c['self_k'], c['self_v'] = c['self_k'][parents], c['self_v'][parents]
                if bool(finished.all()):
                    break
            best = torch.arange(n, device=device) * k + scores.argmax(-1)
            located_history = located_history[best]
            # units of the input height -> fractions of the crop's own input width / height
            located_history = torch.stack([located_history[..., 0] * imgs.shape[2] / widths[:, None],
                                           located_history[..., 1]], dim=-1).tolist()
            history = history[best].tolist()
        results = []
        for seq, seq_locs in zip(history, located_history):
            kept = [(t, None if math.isnan(loc[0]) else tuple(loc)) for t, loc in zip(seq, seq_locs)
                    if t not in (tok.EOS, tok.PAD)]
            results.append(self._to_decoded([t for t, _ in kept], [loc for _, loc in kept]))
        return results

    def _to_decoded(self, seq: List[int], locs: List[Optional[CropLoc]]) -> Decoded:
        tok = self.tokenizer
        text, offsets = [], {}
        position = 0
        for token_id, loc in zip(seq, locs):
            if token_id == tok.SOS:
                continue
            token = tok.idx2token.get(token_id, '')
            if loc is not None:
                offsets[position] = loc
            text.append(token)
            position += len(token)
        return Decoded("".join(text), offsets)

    def predict(self, pages: List[DatabasePage],
                callback: Optional[PredictionCallback] = None) -> AlgorithmPredictionResultGenerator:
        progress = PredictionProgress(callback, len(pages))
        progress.start()
        for db_page in pages:
            pcgts = db_page.pcgts()
            page = pcgts.page
            annotations = Annotations(page)
            block_results = []

            try:
                full_image = Image.open(db_page.file("color_original").local_path())
            except Exception as e:
                logger.warning(f"Skipping page {db_page.page}: could not open image ({e})")
                progress.page_finished()
                yield PredictionResult(pcgts=pcgts, dataset_page=db_page, blocks=[], text_lines=[],
                                       annotations=annotations)
                continue

            music_blocks = [b for b in page.blocks_of_type([BlockType.MUSIC]) if len(b.lines) > 0]
            text_regions = [find_text_region_for_music_block(page, block) for block in music_blocks]
            boxes = [compute_crop_box(page, block, region) for block, region in zip(music_blocks, text_regions)]
            decoded_blocks = self._predict_crops([full_image.crop(box) for box in boxes]) if boxes else []
            # lyric line id -> (lyrics block, line, [(order, music block, syllable chunk)]); a line can
            # collect syllables of several music blocks when the music line is split but the lyrics are not
            lyrics: Dict[str, Tuple[Block, Line, List[Tuple[Tuple[float, int], Block, Chunk]]]] = {}
            for block, text_region, box, decoded in zip(music_blocks, text_regions, boxes, decoded_blocks):
                music_line = block.lines[0]
                chunks = parse_chunks(decoded.text, class_specs=self.class_specs, locs=decoded.locs)
                symbols = assign_symbol_coords(chunks, music_line, page, box)
                block_results.append(SingleBlockResult(music_line=music_line, symbols=symbols, chunks=chunks))

                candidates = lyric_lines_for_music_block(page, block, text_region)
                for region, line in candidates:
                    lyrics.setdefault(line.id, (region, line, []))
                if candidates:
                    syllable_chunks = [c for c in chunks if c.text]
                    targets = assign_chunks_to_lines(chunks, [line for _, line in candidates],
                                                     START_TOLERANCE * block.aabb.height())
                    for i, (chunk, target) in enumerate(zip(syllable_chunks, targets)):
                        lyrics[candidates[target][1].id][2].append(((block.aabb.left(), i), block, chunk))

            text_lines = []
            drop_capitals = page.blocks_of_type([BlockType.DROP_CAPITAL])
            for region, line, entries in lyrics.values():
                entries.sort(key=lambda e: e[0])
                text_lines += lyric_line_results(region, line, [(b, c) for _, b, c in entries], annotations,
                                                 drop_capitals)

            progress.page_finished()
            yield PredictionResult(pcgts=pcgts, dataset_page=db_page, blocks=block_results, text_lines=text_lines,
                                   annotations=annotations)
