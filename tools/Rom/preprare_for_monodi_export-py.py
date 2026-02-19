import enum
import os

from database import DatabaseBook
from database.file_formats.pcgts.page import Sentence

class Manuscript(enum.IntEnum):
    ROM3 = 0
    ROM4 = 1
    Rom5 = 2
    GREG = 3
    MIL = 4

if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    book = DatabaseBook("Rom_1_export")
    pages = book.pages()
    lines = None
    manuscript = Manuscript.GREG
    with open("groupings.csv", "r") as csv_file:
        lines_ = csv_file.readlines()
        lines = [line.strip() for line in lines_]

    excepted_ids = []
    for page, line in list(zip(pages, lines)):
        page_id = page.page
        print(page_id)
        blocks= page.pcgts().page.blocks
        lyric_lines = sorted([(l, l.lines[0]) for l in blocks if l.block_type == l.block_type.LYRICS]
                                , key=lambda x: x[1].coords.aabb().tl.y)
        music_lines = sorted([(l, l.lines[0]) for l in blocks if l.block_type == l.block_type.MUSIC],
                                key=lambda x: x[1].coords.aabb().tl.y)
        total_lines = 0
        keep= []
        for i in line.split(","):
            i = i.strip()
            t= None
            if "+" in i:
                t= i[0]
            else:
                t = i
            length = int(t)
            if t == "0":
                continue

            lines1 = list(zip(lyric_lines, music_lines))
            lyric_lines_group = lines1[total_lines:total_lines + length]
            if manuscript == Manuscript.ROM3 and  "+" in i:
                keep.append(lyric_lines_group[0])
            elif manuscript == Manuscript.ROM3 and "+" not in i:
                continue

            if manuscript != Manuscript.ROM3 and "+" in i:
                keep.append(lyric_lines_group[manuscript.value])
            else:
                select = manuscript.value - 1
                keep.append(lyric_lines_group[select])
            total_lines += length

        blocks_to_keep = [l[0][0].id for l in keep] + [l[1][0].id for l in keep]
        for ind, x in reversed(list(enumerate(page.pcgts().page.blocks))):
            if x.id in blocks_to_keep:
                continue
            else:
                print(f"Deleting block {x.block_type} at index {ind}")
                del page.pcgts().page.blocks[ind]

        page.pcgts().to_file(page.file('pcgts').local_path())