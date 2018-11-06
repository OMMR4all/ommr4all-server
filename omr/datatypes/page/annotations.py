from typing import List
from . import *

from omr.datatypes.page import page as dt_page

class NeumeConnector:
    def __init__(self,
                 neume: Neume,
                 ):
        self.neume = neume
        assert(self.neume is not None)

    @staticmethod
    def from_json(json: dict, mr: MusicRegion):
        return NeumeConnector(
            mr.neume_by_id(json['refID'])
        )

    def to_json(self):
        return {
            'refID': self.neume.id
        }


class SyllableConnector:
    def __init__(self,
                 syllable: Syllable,
                 neume_connections: List[NeumeConnector],
                 ):
        self.syllable = syllable
        self.neume_connections = neume_connections if neume_connections else []
        assert(self.syllable is not None)

    @staticmethod
    def from_json(json: dict, mr: MusicRegion, tr: TextRegion):
        return SyllableConnector(
            tr.syllable_by_id(json['refID']),
            [NeumeConnector.from_json(nc, mr) for nc in json['neumeConnectors']],
        )

    def to_json(self):
        return {
            'refID': self.syllable.id,
            'neumeConnectors': [nc.to_json() for nc in self.neume_connections],
        }


class Connection:
    def __init__(self,
                 music_region: MusicRegion = None,
                 text_region: TextRegion = None,
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
        return Connection(
            mr, tr,
            [SyllableConnector.from_json(s, mr, tr) for s in json['syllableConnectors']]
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
                           [Connection.from_json(c, page) for c in json['connections']]
                           )

    def to_json(self) -> dict:
        return {
            'connections': [c.to_json() for c in self.connections]
        }

    def lyrics_of_music_line(self, mr: MusicRegion) -> TextRegion:
        for connection in self.connections:
            if connection.music_region == mr:
                return connection.text_region

