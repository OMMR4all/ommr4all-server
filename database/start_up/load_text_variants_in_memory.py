import json
import os

from ommr4all import settings
from tools.extract_full_text import get_csv_text
from tools.simple_gregorianik_text_export import Lyrics

lyrics = None

def load_model():
    global lyrics
    path =  os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection', 'lyrics_by_sources.json')

    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    print("loaded latine gr")
    #with open("latine_collection_an.json") as f:
    #    json2 = json.load(f)
    #    lyrics2 = Lyrics.from_dict(json2)
    #print("loaded latine an")

    #lyrics3 = get_csv_text("/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/chant.csv")
    #print("loaded cantus csv")

    #lyrics = Lyrics(lyrics.lyrics + lyrics2.lyrics + lyrics3.lyrics)

def get_data():
  if lyrics is None:
      raise Exception("Expensive model not loaded")
  return lyrics