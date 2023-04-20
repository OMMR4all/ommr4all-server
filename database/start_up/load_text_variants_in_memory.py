import json
import os

from ommr4all import settings
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics
from loguru import logger

lyrics = None
syllable_dictionary = None

def load_model():
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
