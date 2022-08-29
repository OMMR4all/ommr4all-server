from . import *

from database.file_formats.pcgts.page import page as dt_page, Syllable
from typing import List
import logging


logger = logging.getLogger(__name__)


class SyllableConnector:
    def __init__(self,
                 syllable: Syllable,
                 note: MusicSymbol,
                 ):
        self.note = note
        self.syllable = syllable
        assert(self.syllable is not None)
        assert(self.note is not None)
        assert(self.note.symbol_type == SymbolType.NOTE)

    @staticmethod
    def from_json(json: dict, mr: Block, tr: Block):
        try:
            return SyllableConnector(
                tr.syllable_by_id(json['syllableID'], True),
                mr.note_by_id(json['noteID'], True)
            )
        except ValueError as e:
            logger.exception(e)
            return None

    def to_json(self):
        return {
            'syllableID': self.syllable.id,
            'noteID': self.note.id,
        }


class Connection:
    def __init__(self,
                 music_region: Block = None,
                 text_region: Block = None,
                 syllable_connections: List[SyllableConnector] = None,
                 ):
        self.music_region = music_region
        self.text_region = text_region
        self.syllable_connections: List[SyllableConnector] = syllable_connections if syllable_connections else []
        assert(self.music_region is not None)
        assert(self.text_region is not None)

    @staticmethod
    def from_json(json: dict, page: dt_page.Page):
        mr = page.music_region_by_id(json['musicID'])
        tr = page.text_region_by_id(json['textID'])
        if not mr or not tr:
            logger.warning("Invalid music or text region ({}/{})".format(mr, tr))
            return

        return Connection(
            mr, tr,
            [s for s in [SyllableConnector.from_json(s, mr, tr) for s in json['syllableConnectors']] if s]
        )

    def to_json(self):
        return {
            'musicID': self.music_region.id,
            'textID': self.text_region.id,
            'syllableConnectors': [s.to_json() for s in self.syllable_connections],
        }


class Annotations:
    def __init__(self, page: dt_page.Page, connections: List[Connection] = None):
        self.page = page
        self.connections: List[Connection] = connections if connections else []
        assert(self.page is not None)

    @staticmethod
    def from_json(json: dict, page: dt_page.Page):
        return Annotations(page,
                           [c for c in [Connection.from_json(c, page) for c in json['connections']] if c]
                           )

    def to_json(self) -> dict:
        return {
            'connections': [c.to_json() for c in self.connections]
        }

    def drop_annotation_by_text_block(self, block: Block):
        self.connections = [c for c in self.connections if c.text_region != block]

    def lyrics_of_music_line(self, mr: Block) -> Block:
        for connection in self.connections:
            if connection.music_region == mr:
                return connection.text_region


    def get_symbols_of_line(self, mr: Block, line: Line) -> List[MusicSymbol]:
        for connection in self.connections:
            if connection.music_region == mr:
                for line in mr.lines:
                    pass
    def get_or_create_connection(self, m: Block, t: Block) -> Connection:
        for c in self.connections:
            if c.music_region == m and c.text_region == t:
                return c

        c = Connection(m, t)
        self.connections.append(c)
        return c
