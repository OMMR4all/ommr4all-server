import csv

from tools.simple_gregorianik_text_export import Lyrics, Lyric_info


def get_csv_text(csv_path):
    chants = []
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
                continue
            chant = row[18]
            chant.replace("-", "").replace("*","").replace("(","").replace(")", "")
            if len(row[18]) > 10:
                chants.append(Lyric_info(id=None, index = None, meta_info = None, meta_infos_extended =None, variants=None, latine=row[18]))
    return Lyrics(chants)
if __name__ == "__main__":
    csv_path = "/home/alexanderh/Pictures/cantuscorpus-v0.2/csv/chant.csv"
    get_csv_text(csv_path)
