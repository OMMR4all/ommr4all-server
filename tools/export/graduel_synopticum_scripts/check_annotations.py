from typing import List

import numpy as np
from PIL import Image, ImageDraw
from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts import PageScaleReference, MusicSymbol, Page

book = DatabaseBook('Graduel_Syn22_03_24')
documents = DatabaseBookDocuments().load(book)


def draw_symbols(image: ImageDraw, symbols: List[MusicSymbol], color=(0, 0, 255), page: Page = None):
    for s in symbols:
        s: MusicSymbol
        coord = page.page_to_image_scale(s.coord, PageScaleReference.NORMALIZED_X2)

        image.rectangle((coord.x - 5, coord.y - 5, coord.x + 5, coord.y + 5,), outline=color, width=2)

def move_syllable_to_neume_start(syllable, page, music_region, connection):
    symbols = music_region.lines[0].symbols
    note = connection.note
    index = symbols.index(note)
    def prev_symbol():
        for i in range(index - 1, -1, -1):
            if symbols[i].symbol_type == symbols[i].symbol_type.NOTE and symbols[i].graphical_connection == symbols[i].graphical_connection.NEUME_START:
                return symbols[i]
        return None
    prev_symbol = prev_symbol()
    return prev_symbol
count = 0
for i in book.pages():
    for t in i.pcgts().page.annotations.connections:
        for f in t.syllable_connections:
            if f.note.graphical_connection != f.note.graphical_connection.NEUME_START:
                image = Image.open(i.pcgts().page.location.file("color_norm_x2").local_path())

                print(i.pcgts().page.location.page)
                print(f.note.id)
                print(f.note.graphical_connection)
                print(t.music_region.id)
                note = move_syllable_to_neume_start(f.syllable, i.pcgts().page, t.music_region, f)
                count += 1
                draw_symbols(ImageDraw.Draw(image), [f.note], color=(255, 0, 0), page=i.pcgts().page)
                draw_symbols(ImageDraw.Draw(image), [note], color=(0, 255, 0), page=i.pcgts().page)

                np_image = np.array(image)
                from matplotlib import pyplot as plt
                #plt.imshow(np_image)
                #plt.show()
                f.note = note
    i.pcgts().to_file(i.file('pcgts').local_path())

print(count)