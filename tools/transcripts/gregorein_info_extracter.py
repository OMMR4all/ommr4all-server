import json
import os
import urllib.request
import requests
from bs4 import BeautifulSoup
from lxml import etree

from database.file_formats.book.document import DatasetSource
from ommr4all import settings
from tools.simple_gregorianik_text_export import Lyric_info, Lyrics


def simple_extract():
    baseurl = "https://gregorien.info/chant/id/{}/0/de"
    lyrics = []
    for i in range(1, 8810):
        print(i)
        url = baseurl.format(i)
        page = requests.get(url)
        if not page.ok:
            print(f"not available {i}")
            continue
        contents = page.content
        soup = BeautifulSoup(contents, 'html.parser')
        sections = soup.find_all("section")
        if len(sections) <3:
            continue
        print(i)

        genre = sections[0].find_all("h3")[1].find("span").get_text()
        latine = sections[1].find_all("div")[1].find("div").get_text().replace("\n", " ").replace("\r", "")
        links = sections[2].find_all("a")
        cantus_id = None
        for t in links:
            if t.get_text() == "Cantus Index":
                href= t["href"]
                cantus_id = href.split("/")[-1]



        initium = " ".join(latine.split(" ")[:3])
        meta_infos = []
        variants = []
        meta_info = ""
        lyric_info = Lyric_info(index=str(i), id=i, meta_info=meta_info, latine=latine, variants=variants,
                                meta_infos_extended=meta_infos, genre=genre, initium=initium, url=url, cantus_id=cantus_id, dataset_source=DatasetSource.GI)
        # print(lyric_info.to_json())
        lyrics.append(lyric_info)
    return lyrics


if __name__ == "__main__":
    path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                        'lyrics_by_sources.json')


    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    lyric_info1 = lyrics.lyrics

    lyric_info2 = simple_extract()

    lyric_extended = Lyrics(lyric_info1 + lyric_info2)
    with open('lyrics_by_sources.json', 'w', encoding='utf-8') as f:
        json.dump(lyric_extended.to_dict(), f, ensure_ascii=False, indent=4)