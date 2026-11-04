"""Cheap structural facts about a page, derived from its pcgts without parsing it.

The batch predicates of the algorithm steps (`AlgorithmPredictor.unprocessed` and
`.runnable`) each need one or two counts: "does the page have music blocks", "does any
music line carry symbols", "is a lyric line still without syllables". Answering them
from a full `PcGts.from_file` costs ~90 ms on a page of Moosburger_Graduale, which a
whole-book page selection multiplies by the size of the book -- 47 s and several GiB of
parsed object graph for its 532 pages, enough for the dev server to kill the request
while the workflow tab was only asking for a page count.

`PageContent` is what those predicates actually read. It is computed from the raw
pcgts json (no geometry, no symbol objects) and cached in the page index keyed by the
pcgts mtime, so a page selection answers them from one DB query plus a stat per page.
The counts mirror what the parsed page would report, so the predicates keep their
exact semantics -- `from_page` is the authority for that and is checked against
`from_pcgts_json` in the tests.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING, Union

from database.file_formats.pcgts.page.definitions import BlockType

if TYPE_CHECKING:
    from database.file_formats.pcgts.page.page import Page

# bumped whenever the computed fields change; a row written by an older version is
# recomputed instead of being read back with fields that are missing or mean something else
VERSION = 1

BlockTypes = Union[BlockType, Iterable[BlockType]]

# what Page.text_blocks() returns: everything that is not music
TEXT_BLOCK_TYPES: List[BlockType] = [t for t in BlockType if t != BlockType.MUSIC]

MUSIC = BlockType.MUSIC.value


def _keys(block_type: BlockTypes) -> List[str]:
    if isinstance(block_type, BlockType):
        return [block_type.value]
    return [t.value for t in block_type]


@dataclass(frozen=True)
class PageContent:
    """Structural counts of one page. All dicts are keyed by BlockType value."""
    blocks: Dict[str, int] = field(default_factory=dict)        # type -> number of blocks
    lines: Dict[str, int] = field(default_factory=dict)         # type -> number of lines
    lines_without_syllables: Dict[str, int] = field(default_factory=dict)
    music_lines_with_symbols: int = 0
    music_lines_with_staff_lines: int = 0
    connections: int = 0

    def n_blocks(self, block_type: BlockTypes) -> int:
        return sum(self.blocks.get(k, 0) for k in _keys(block_type))

    def n_lines(self, block_type: BlockTypes) -> int:
        return sum(self.lines.get(k, 0) for k in _keys(block_type))

    def n_lines_without_syllables(self, block_type: BlockTypes) -> int:
        return sum(self.lines_without_syllables.get(k, 0) for k in _keys(block_type))

    @property
    def n_music_blocks(self) -> int:
        return self.n_blocks(BlockType.MUSIC)

    @property
    def n_text_blocks(self) -> int:
        return self.n_blocks(TEXT_BLOCK_TYPES)

    @property
    def has_music_symbols(self) -> bool:
        return self.music_lines_with_symbols > 0

    @property
    def has_staff_lines(self) -> bool:
        return self.music_lines_with_staff_lines > 0

    def to_dict(self) -> dict:
        return {
            'version': VERSION,
            'blocks': self.blocks,
            'lines': self.lines,
            'lines_without_syllables': self.lines_without_syllables,
            'music_lines_with_symbols': self.music_lines_with_symbols,
            'music_lines_with_staff_lines': self.music_lines_with_staff_lines,
            'connections': self.connections,
        }

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional['PageContent']:
        """The stored counts, or None if they are missing or were written by an older version
        (the caller then recomputes them from the pcgts)."""
        if not isinstance(d, dict) or d.get('version') != VERSION:
            return None
        return PageContent(
            blocks=d.get('blocks') or {},
            lines=d.get('lines') or {},
            lines_without_syllables=d.get('lines_without_syllables') or {},
            music_lines_with_symbols=d.get('music_lines_with_symbols', 0),
            music_lines_with_staff_lines=d.get('music_lines_with_staff_lines', 0),
            connections=d.get('connections', 0),
        )

    @staticmethod
    def from_page(page: 'Page') -> 'PageContent':
        """Counts of an already parsed page -- the definition the json version mirrors."""
        blocks: Dict[str, int] = {}
        lines: Dict[str, int] = {}
        without_syllables: Dict[str, int] = {}
        with_symbols = with_staff_lines = 0

        for block in page.blocks:
            key = block.block_type.value
            blocks[key] = blocks.get(key, 0) + 1
            lines[key] = lines.get(key, 0) + len(block.lines)
            for line in block.lines:
                if block.block_type == BlockType.MUSIC:
                    if line.symbols:
                        with_symbols += 1
                    if line.staff_lines:
                        with_staff_lines += 1
                if not line.sentence.syllables:
                    without_syllables[key] = without_syllables.get(key, 0) + 1

        return PageContent(blocks, lines, without_syllables,
                           with_symbols, with_staff_lines, len(page.annotations.connections))

    @staticmethod
    def from_pcgts_json(d: Optional[dict]) -> 'PageContent':
        """Counts from the raw pcgts json (None -- e.g. a page without a pcgts file -- is empty)."""
        page = (d or {}).get('page') or {}
        blocks: Dict[str, int] = {}
        lines: Dict[str, int] = {}
        without_syllables: Dict[str, int] = {}
        with_symbols = with_staff_lines = 0
        music_ids, text_ids = set(), set()

        for block in page.get('blocks') or []:
            key = block.get('type', MUSIC)  # Block.from_json defaults to music
            blocks[key] = blocks.get(key, 0) + 1
            if block.get('id'):
                (music_ids if key == MUSIC else text_ids).add(block['id'])
            block_lines = block.get('lines') or []
            lines[key] = lines.get(key, 0) + len(block_lines)
            for line in block_lines:
                if key == MUSIC:
                    if line.get('symbols'):
                        with_symbols += 1
                    if line.get('staffLines'):
                        with_staff_lines += 1
                # Sentence.from_json drops syllables without text, so they do not count here either
                syllables = (line.get('sentence') or {}).get('syllables') or []
                if not any(s.get('text') for s in syllables):
                    without_syllables[key] = without_syllables.get(key, 0) + 1

        # Connection.from_json drops a connection whose music or text block is gone
        connections = sum(1 for c in (page.get('annotations') or {}).get('connections') or []
                          if c.get('musicID') in music_ids and c.get('textID') in text_ids)

        return PageContent(blocks, lines, without_syllables,
                           with_symbols, with_staff_lines, connections)
