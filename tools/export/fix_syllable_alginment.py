from typing import List

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts import MusicSymbol
from database.file_formats.pcgts.page import Sentence

def get_previous_note_start_symbol(symbols: List[MusicSymbol], index: int):
    c_index = index
    while c_index > 0:
        c_index = c_index - 1
        c_symbol = symbols[c_index]
        if c_symbol.graphical_connection == c_symbol.graphical_connection.NEUME_START:
            return c_symbol
    return symbols[index]

if __name__ == "__main__":

    book = DatabaseBook('Graduel_Syn2')

    pages = book.pages()
    for i in pages:
        page = i.pcgts().page
        annotation = page.annotations
        for con in annotation.connections:
            prev_syl_con = None
            for syl_con in con.syllable_connections:
                if syl_con.note.graphical_connection != syl_con.note.graphical_connection.NEUME_START:
                    symbols = con.music_region.lines[0].symbols
                    n_symbol = get_previous_note_start_symbol(symbols, symbols.index(syl_con.note))
                    if prev_syl_con is None or prev_syl_con.note != n_symbol:
                        syl_con.note = n_symbol
                prev_syl_con = syl_con

        i.pcgts().to_file(i.file('pcgts').local_path())


