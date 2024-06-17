import os

from database import DatabaseBook, DatabasePage


def fix_custodes(page: DatabasePage, lastsymbol=None, prevline=None, percentage=0.01):
    lines = page.pcgts().page.all_music_lines()
    last_symbol = lastsymbol
    prev_line = prevline
    if page.page == "folio_0202v":
        print("asds")
        pass
    for ind, i in enumerate(lines):
        text_line = page.pcgts().page.closest_below_text_line_to_music_line(i)
        if text_line.document_start:
            last_symbol = None
            prev_line = None
            #prev_line_page = None
        symbols = [t for t in i.symbols if t.symbol_type == t.symbol_type.NOTE]
        if len(symbols) == 0:
            continue
        l_symbol = symbols[-1]

        if last_symbol and symbols[0].note_name == last_symbol.note_name:
            #if last_symbol.graphical_connection == last_symbol.graphical_connection.NEUME_START:
            del prev_line.symbols[-1]

        if abs(l_symbol.coord.x - i.coords.aabb().br.x) < percentage:
            #if l_symbol.graphical_connection == l_symbol.graphical_connection.NEUME_START:
            last_symbol = l_symbol
            prev_line = i
            #else:
            #    last_symbol = None
            #    prev_line = None
        else:
            last_symbol = None
            prev_line = None
    return last_symbol, prev_line


if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings

    # get all books from database
    pages = DatabaseBook("mul_2_end_w_finetune_basic_w_doc_pp_w_symbolpp").pages()
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files

    for page in pages:
        page_id = page.page
        fix_custodes(page)

    for page in pages:
        page.pcgts().to_file(page.file('pcgts').local_path())
