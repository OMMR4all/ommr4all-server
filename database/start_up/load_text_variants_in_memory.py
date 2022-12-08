import json
import os

from ommr4all import settings
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics
from loguru import logger

lyrics = None


def load_model():
    global lyrics
    path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                        'lyrics_by_sources.json')
    assert os.path.exists(path)

    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    logger.info("Successfully imported Lyrics database into memory")


def get_data():
    if lyrics is None:
        raise Exception("Expensive model not loaded")
    return lyrics
