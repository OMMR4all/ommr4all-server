import json
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Metainfo:
    festum: str
    dies: str
    versus: str
    sources: List[str]


@dataclass_json
@dataclass
class Variant:
    source: str
    latine: str


@dataclass_json
@dataclass
class Lyric_info:
    index: str
    id: str
    latine: str
    meta_info: str
    meta_infos_extended: List[Metainfo]
    variants: List[Variant]


@dataclass_json
@dataclass
class Lyrics:
    lyrics: List[Lyric_info]


def web_page_b_index(index: int):
    page = requests.get("http://gregorianik.uni-regensburg.de/an/view/detail?ref={}".format(index))
    if not page.ok:
        return None
    contents = page.content
    soup = BeautifulSoup(contents, 'html.parser')

    latine = soup.find_all('p')[0].get_text().strip().replace("\n", " ")
    meta_info1 = soup.find_all("div", {"class": "topbar"})
    meta_info = ""
    for i in meta_info1:
        meta_info += " ".join(i.get_text().strip().replace("\n", " ").replace("\t", " ").split())
    meta_info = meta_info.strip().replace("\n", " ")
    id = ""
    for i in soup.find_all('h1')[0]:
        id += i.get_text()
    id = id.strip().replace("\n", " ")

    variants = []
    for i in soup.find_all("div", {"class": "text"}):
        variants.append(Variant(source=i.find_previous_sibling().get_text().strip(),
                                latine=i.get_text().strip().replace("\n", " ")))

    meta_infos = []
    meta_table = soup.find("table", {"class": "litdates"})
    for i in meta_table.find_all("tr")[1:]:
        festum = i.find_all("td")[0].get_text().strip().replace("\n", " ")
        dies = i.find_all("td")[1].get_text().strip().replace("\n", " ")
        versus = i.find_all("td")[2].get_text().strip().replace("\n", " ")
        sources = []
        for i in i.find_all("td")[3].find_all("li"):
            sources.append(i.get_text().strip().replace("\n", " "))
        meta_infos.append(Metainfo(festum=festum, dies=dies, versus=versus, sources=sources))

    lyric_info = Lyric_info(index=str(index), id=id, meta_info=meta_info, latine=latine, variants=variants,
                            meta_infos_extended=meta_infos)
    return lyric_info


def crawl(start=1, stop=100000, interval=6600):
    lyrics = []
    pass
    json_interval = 1

    for i in range(start, stop):
        a = web_page_b_index(i)
        lyrics.append(a)
        if len(lyrics) > interval:
            lyr = Lyrics(lyrics)
            with open('data_an{}.json'.format(json_interval), 'w', encoding='utf-8') as f:
                json.dump(lyr.to_dict(), f, ensure_ascii=False, indent=4)
            lyrics = []
            json_interval += 1

    lyr = Lyrics(lyrics)
    with open('data_an{}.json'.format(json_interval), 'w', encoding='utf-8') as f:
        json.dump(lyr.to_dict(), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    crawl()
    pass
