from typing import List
from . import *


class NeumeConnector:
    def __init__(self,
                 neume: Neume,
                 ):
        self.neume = neume


class SyllableConnector:
    def __init__(self,
                 syllable: Syllable,
                 neume_connection: NeumeConnector,
                 ):
        self.syllable = syllable
        self.neume_connection = neume_connection


class Connection:
    def __init__(self,
                 music_region: MusicRegion = None,
                 text_region: TextRegion = None,
                 syllable_connections: List[SyllableConnector] = None,
                 ):
        self.music_region = music_region
        self.text_region = text_region
        self.syllable_connections: List[SyllableConnector] = syllable_connections if syllable_connections else []


class Annotations:
    def __init__(self, connections: List[Connection] = None):
        self.connections: List[Connection] = connections if connections else []
