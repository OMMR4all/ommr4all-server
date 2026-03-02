import multiprocessing
import sys
import inspect
import logging



original_import = __import__
#logging.basicConfig(
#    filename='/tmp/import_log.txt',
#    level=logging.INFO,
#    format='%(asctime)s - %(message)s'
#)

imported_modules = []
def find_importer_file():
    for frame_info in inspect.stack():
        if "importlib" not in frame_info.filename:
            return frame_info.filename
    return "Unbekannt"

def find_importer_chain():
    chain = []
    for frame_info in inspect.stack():
        if "importlib" not in frame_info.filename:
            chain.append(frame_info.filename)
    return " -> ".join(chain)

def my_custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name != "torch":
        return module
    importer_file = find_importer_file()
    imported_file = "N/A"
    if hasattr(module, '__file__'):
        imported_file = module.__file__

    logging.info(f"Die Bibliothek '{name}' wurde importiert.")
    logging.info(f"  - Aufgerufen in der Datei: {importer_file}")
    logging.info(f"  - Pfad der importierten Bibliothek: {imported_file}")
    logging.info(f"  - Importkette: {find_importer_chain()}")
    logging.info("-" * 50)

    return module

from ommr4all import settings
from tools.simple_gregorianik_text_export import Lyrics
from loguru import logger

import os
import json
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class LyricsData:
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LyricsData, cls).__new__(cls)
        return cls._instance

    def load(self):
        if self._is_initialized:
            return

        lyrics_path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                                   'lyrics_by_sources.json')
        if os.path.exists(lyrics_path):
            with open(lyrics_path) as f:
                self.lyrics = Lyrics.from_dict(json.load(f))
            logger.info("Successfully imported Lyrics database.")

        dict_path = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_dictionary',
                                 'syllable_dictionary.json')
        if os.path.exists(dict_path):
            with open(dict_path) as f:
                self.syllable_dictionary = json.load(f)
            logger.info("Successfully imported Syllable dictionary.")

        self._is_initialized = True


lyrics_store = LyricsData()
