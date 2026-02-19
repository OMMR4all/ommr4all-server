import json
import os

from database.file_formats.book.document import DatasetSource
from tools.simple_gregorianik_text_export import Lyrics, Lyric_info

if __name__ == '__main__':
    pass
    transcripts_file_path = "transcripts.txt"
    path = 'lyrics_by_sources.json'


    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    lyrics2 = []
    with open(transcripts_file_path) as f:
        for ind, i in enumerate(f.readlines()):

            latine = i
            if len(latine) > 1:
                initium = " ".join(latine.split(" ")[:3])

                lyric_info = Lyric_info(index=str(ind), id=ind, meta_info="", latine=latine, variants=[],
                                        meta_infos_extended=[], genre="", initium=initium, url="",
                                        cantus_id=None, dataset_source=DatasetSource.GE)
                lyrics2.append(lyric_info)
    lyric_info1 = lyrics.lyrics


    lyric_extended = Lyrics(lyric_info1 + lyrics2)
    with open('lyrics_by_sources1.json', 'w', encoding='utf-8') as f:
        json.dump(lyric_extended.to_dict(), f, ensure_ascii=False, indent=4)