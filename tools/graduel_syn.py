import json
from multiprocessing import Pool

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from tools.simple_gregorianik_text_export import get_genre_from_cantus_url, Variant, Metainfo, Lyric_info, Lyrics

b_url = "http://gregorianik.uni-regensburg.de/gr/#id/"
image_url = "http://gregorianik.uni-regensburg.de/gr/tab/0001%20Ytab.png"
def web_page_b_index2(index: int):
    url = "http://gregorianik.uni-regensburg.de/gr/view/detail?ref={}".format(index)
    image_url = f"http://gregorianik.uni-regensburg.de/gr/tab/{index:04d}%20Ytab.png"

    page = requests.get(url)
    if not page.ok:
        return None
    contents = page.content
    soup = BeautifulSoup(contents, 'html.parser')
    image = requests.get(image_url)
    if not image.ok:
        return None

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

    with open(f"/tmp/files/{index}.json",  'w', encoding='utf-8') as f:
        json.dump(lyric_info.to_dict(), f, ensure_ascii=False, indent=4)
    with open(f"/tmp/files/{index}.png", 'wb') as f:
        f.write(image.content)
    # print(lyric_info.to_json())
    return None

def crawl(start=1, stop=4000, interval=7000):
    lyrics = []
    pass
    json_interval = 1
    current_len = 1
    pool = Pool(processes=6)
    indexes = list(range(start, stop))
    lyrics = list(tqdm(pool.imap(web_page_b_index2, indexes), total=len(indexes)))

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
    #lyr = Lyrics(lyrics)
    #with open('data_an{}.json'.format(json_interval), 'w', encoding='utf-8') as f:
    #    json.dump(lyr.to_dict(), f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    crawl()
    pass
