import os

from database import DatabaseBook


def to_file(jsonf, filename):
    if filename.endswith(".json"):
        import json
        # first dump to keep file if an error occurs
        s = json.dumps(jsonf, indent=2)
        with open(filename, 'w') as f:
            f.write(s)
    else:
        raise Exception("Invalid file extension of file '{}'".format(filename))
if __name__ == "__main__":

    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings

    # get all books from database
    pages = DatabaseBook("mul_2_rsync_gt").pages()#fix veriefied
    #pages = DatabaseBook("Pa_14819_gt").pages()
    #pages = DatabaseBook("Koeln_Dombibl_1001b_part_gt").pages()
    #pages = DatabaseBook("Koeln_Dombibl_1001b_part_gt").pages()
    #pages = DatabaseBook("Geesebook2_andreas_gt").pages()
    #pages = DatabaseBook("Graduel_Syn").pages()

    #pages = DatabaseBook("Geesebook1_complete_fixed_ro").pages()
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files

    ind = 0
    bookname = "Mulhouse2"

    path = "/tmp/"
    p_c = os.path.join(path, bookname)
    os.mkdir(p_c)
    for page in pages:
        pp = os.path.join(p_c, page.page)
        os.mkdir(pp)
        if page.page_progress().verified_allowed():
            pcgts1 = page.pcgts()
            js = pcgts1.to_json(True)
            pcgts = "pcgts.json"
            to_file(js, os.path.join(pp, pcgts))
            progress = "page_progress.json"
            page.page_progress().to_json_file(os.path.join(pp, progress))
            meta = "meta.json"
            js2 = page.meta().to_json()
            to_file(js2, os.path.join(pp, meta))