import enum
import json
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

import requests
from bs4 import BeautifulSoup
from dataclasses_json import dataclass_json
from mashumaro.mixins.json import DataClassJSONMixin
from tqdm import tqdm

from database.file_formats.book.document import DatasetSource


@dataclass
class Metainfo(DataClassJSONMixin):
    festum: str
    dies: str
    versus: str
    sources: List[str]


@dataclass
class Variant(DataClassJSONMixin):
    source: str
    latine: str


@dataclass
class Lyric_info(DataClassJSONMixin):
    index: str
    id: str
    latine: str
    meta_info: str
    meta_infos_extended: List[Metainfo]
    variants: List[Variant]
    cantus_id: str = None
    initium: str = None
    genre: str = None
    url: str = None
    dataset_source: DatasetSource = None


@dataclass
class Lyrics(DataClassJSONMixin):
    lyrics: List[Lyric_info]


def get_genre_from_cantus_url(url):
    page = requests.get(url)
    genre = None
    if not page.ok:
        return None
    contents = page.content
    soup = BeautifulSoup(contents, 'html.parser')
    if soup is not None:
        row = soup.find("tr", {"class": "odd views-row-first"})
        if row is not None:
            genre = row.find("td", {"class": "views-field-field-mc-genre"}).get_text()
    return genre


def web_page_b_index(index: int):
    url = "http://gregorianik.uni-regensburg.de/an/view/detail?ref={}".format(index)
    page = requests.get(url)
    if not page.ok:
        return None
    contents = page.content
    soup = BeautifulSoup(contents, 'html.parser')

    latine = soup.find_all('p')[0].get_text().strip().replace("\n", " ")
    meta_info1 = soup.find("div", {"class": "topbar"})

    meta_info = ""
    meta_info += " ".join(meta_info1.get_text().strip().replace("\n", " ").replace("\t", " ").split())
    # print(meta_info1)
    cantus_content = meta_info1.find("a")
    if cantus_content:
        cantus_id = cantus_content.get_text()
        cantus_url = cantus_content['href']
        genre = get_genre_from_cantus_url(cantus_url)
        # print(genre)

    else:
        genre = soup.find("span", {"class": "forma"}).get_text()
        # print(genre)
        cantus_id = None
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

    initium = soup.find("div", {"class": "detail_content"}).find("h1")
    initium = initium.contents[-1].get_text()

    for i in meta_table.find_all("tr")[1:]:
        festum = i.find_all("td")[0].get_text().strip().replace("\n", " ")
        dies = i.find_all("td")[1].get_text().strip().replace("\n", " ")
        versus = i.find_all("td")[2].get_text().strip().replace("\n", " ")
        sources = []
        for i in i.find_all("td")[3].find_all("li"):
            sources.append(i.get_text().strip().replace("\n", " "))
        meta_infos.append(Metainfo(festum=festum, dies=dies, versus=versus, sources=sources))

    lyric_info = Lyric_info(index=str(index), id=id, meta_info=meta_info, latine=latine, variants=variants,
                            meta_infos_extended=meta_infos, genre=genre, initium=initium, url=url, cantus_id=cantus_id)
    # print(lyric_info.to_json())
    return lyric_info


def crawl(start=1, stop=6600, interval=7000):
    lyrics = []
    pass
    json_interval = 1
    current_len = 1
    pool = Pool(processes=6)
    indexes = list(range(start, stop))
    lyrics = list(tqdm(pool.imap(web_page_b_index, indexes), total=len(indexes)))
    # print(Lyrics(lyrics).to_json())
    """
    for i in tqdm(range(start, stop)):
        a = web_page_b_index(i)

        if a is not None:
            # return a
            lyrics.append(a)
            if current_len > interval:
                lyr = Lyrics(lyrics)
                with open('data_an{}.json'.format(json_interval), 'w', encoding='utf-8') as f:
                    json.dump(lyr.to_dict(), f, ensure_ascii=False, indent=4)
                lyrics = []
                json_interval += 1
        current_len += 1
    """
    lyr = Lyrics(lyrics)
    with open('data_an{}.json'.format(json_interval), 'w', encoding='utf-8') as f:
        json.dump(lyr.to_dict(), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    crawl()
    pass
