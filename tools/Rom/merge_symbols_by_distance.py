from typing import List

import numpy as np

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts import MusicSymbol, PageScaleReference
from database.file_formats.pcgts.page import Sentence

def get_previous_note_start_symbol(symbols: List[MusicSymbol], index: int):
    c_index = index
    while c_index > 0:
        c_index = c_index - 1
        c_symbol = symbols[c_index]
        if c_symbol.graphical_connection == c_symbol.graphical_connection.NEUME_START and c_symbol.symbol_type == c_symbol.symbol_type.NOTE:
            return c_symbol
    return symbols[index]

if __name__ == "__main__":

    book = DatabaseBook('Rom_1')
    def scale(x):
        return np.round(page.page_to_image_scale(x, PageScaleReference.NORMALIZED_X2)).astype(int)

    pages = book.pages()
    for i in pages:
        page = i.pcgts().page
        avg_line_distance = page.avg_staff_line_distance()
        avg_line_distance = scale(avg_line_distance)

        annotation = page.annotations
        change = True
        while change:
            change = False
            for con in annotation.connections:
                notes = [i.note for i in con.syllable_connections]
                #prev_syl_con = None
                for syl_con in con.syllable_connections:
                    symbols = con.music_region.lines[0].symbols
                    n_symbol = get_previous_note_start_symbol(symbols, symbols.index(syl_con.note))
                    if syl_con.note != n_symbol:
                        s1 = scale(syl_con.note.coord)
                        s2 = scale(n_symbol.coord)
                        if abs(s1.x - s2.x) < avg_line_distance*2:
                            if n_symbol in notes:
                                continue

                            syl_con.note = n_symbol
                            change = True

        i.pcgts().to_file(i.file('pcgts').local_path())


