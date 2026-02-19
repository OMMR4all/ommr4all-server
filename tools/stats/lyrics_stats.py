import json
import os

import dataclasses
from collections import defaultdict

from ommr4all import settings
from tools.simple_gregorianik_text_export import Lyrics

path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                    'lyrics_by_sources.json')
assert os.path.exists(path)

path2 = os.path.join(settings.BASE_DIR, 'internal_storage', 'default_dictionary', 'syllable_dictionary.json')
assert os.path.exists(path)
with open(path) as f:
    json1 = json.load(f)
    lyrics = Lyrics.from_dict(json1)

source = defaultdict(list)

for i in lyrics.lyrics:
    source[i.dataset_source].append(i)

for i in source.keys():
    genre = defaultdict(list)
    for t in source[i]:
        genre[t.genre].append(i)
    for t in genre.keys():
        print(t)
        print(len(genre[t]))

