import os

from database import DatabaseBook
from database.file_formats.pcgts.page import Sentence

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    book = DatabaseBook("Rom_1")
    pages = book.pages()
    lines = None
    with open("groupings.csv", "r") as csv_file:
        lines_ = csv_file.readlines()
        lines = [line.strip() for line in lines_]
    # pages = [pages[5]]  # 0:45
    excepted_ids = []
    for page, line in zip(pages, lines):
        page_id = page.page
        #print(line)
        lyric_lines = page.pcgts().page.all_text_lines()
        total_lines = 0
        for i in line.split(","):
            lyric_line = 3
            i = i.strip()
            if "+" in i:
                lyric_line += 1
                i= i[0]
            length = int(i)
            if i == "0":
                continue
            lyric_lines_group = lyric_lines[total_lines:total_lines + length]

            #if page_id == "02_14_ppm_f__ppm__r":
            #    pass

            lyric = lyric_lines_group[lyric_line-1].text()

            for ind, lyric_line_g in enumerate(lyric_lines_group):
                if ind == lyric_line - 1:
                    continue
                else:
                    lyric_line_g.sentence = Sentence.from_string(lyric)
                    lyric_line_g.document_start = False
            total_lines += length

        page.pcgts().to_file(page.file('pcgts').local_path())