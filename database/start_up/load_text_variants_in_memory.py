import multiprocessing
import sys
import inspect
import logging
lyrics = None
syllable_dictionary = None


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
import json
import os

from ommr4all import settings
from tools.simple_gregorianik_text_export import Lyrics
from loguru import logger


def load_model():
    #try:
    #    multiprocessing.set_start_method('spawn')
    #except:
    #    pass

    global lyrics
    global syllable_dictionary
    path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                        'lyrics_by_sources.json')
    assert os.path.exists(path)

    path2 = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_dictionary', 'syllable_dictionary.json')
    assert os.path.exists(path)
    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    logger.info("Successfully imported Lyrics database into memory")

    with open(path2) as f:
        json1 = json.load(f)
        syllable_dictionary = json1
    logger.info("Successfully imported Syllable dictionary into memory")


def get_data():
    if lyrics is None:
        raise Exception("Expensive model not loaded")
    return lyrics
