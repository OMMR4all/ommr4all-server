import os
import json
import loguru
from database.database_dictionary import DatabaseDictionary
from ommr4all import settings
from tools.simple_gregorianik_text_export import Lyrics


if __name__ == "__main__":
    path = "/home/alexanderh/projects/ommr4all3.8transition/ommr4all-deploy/modules/data/default_dictionary.json"

    with open(path) as f:
        json1 = json.load(f)
        lyrics = DatabaseDictionary.from_json(json1)
        lyrics.to_hyphen_dict()
