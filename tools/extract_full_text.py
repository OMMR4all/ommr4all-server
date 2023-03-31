import csv

from tools.simple_gregorianik_text_export import Lyrics, Lyric_info, Metainfo


def genre_dict():
    csv_path = "/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/genre.csv"
    g_dict = {}
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            g_dict[row[0]] = row[1]
    return g_dict


def feast_dict():
    csv_path = "/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/feast.csv"
    f_dict = {}
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            f_dict[row[0]] = row[1]
    return f_dict


def normalize(s: str):
    return s.replace("-", "").replace("*", "").replace("(", "").replace(")", "").replace("*", "")


def get_csv_text(csv_path):
    g_dict = genre_dict()
    f_dict = feast_dict()
    chants = []
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            chant = row[18]
            chant = chant.replace("-", "").replace("*", "").replace("(", "").replace(")", "")

            # initium#
            initium = row[1]
            normaized_initium = normalize(initium)
            if "-" not in initium and "*" not in initium and len(normaized_initium.split(" ")) >= 3:
                initium = normaized_initium
            else:
                initium = None
            #print(row[13])
            #print(g_dict)
            genre = g_dict[row[13]] if row[13] else ""
            feast = f_dict[row[12]] if row[12] else ""
            url = row[17]
            cantusid = url.split("/")[-2]
            # cantusid = row[2]
            if len(row[18]) > 10:
                l_info = Lyric_info(id=None, index=None, meta_info=None, meta_infos_extended=[Metainfo(festum=feast, dies=None, versus=None, sources=None)], variants=None,
                                         latine=row[18], initium=initium, cantus_id=cantusid, genre=genre, url=url)
                chants.append(l_info)
                #print(l_info)
    return Lyrics(chants)

"""
class Metainfo:
    festum: str
    dies: str
    versus: str
    sources: List[str]

"""
if __name__ == "__main__":
    csv_path = "/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/chant.csv"
    get_csv_text(csv_path)
