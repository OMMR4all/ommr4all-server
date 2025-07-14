from database import DatabaseBook

if __name__ == "__main__":

    book = DatabaseBook('Geesebook2gt_gapped_ignore')
    pages = book.pages()
    for i in pages:
        page = i.pcgts().page
        for l in page.all_music_lines():
            for s in l.symbols:
                if s.graphical_connection == s.graphical_connection.GAPED:
                    s.graphical_connection = s.graphical_connection.NEUME_START


        i.pcgts().to_file(i.file('pcgts').local_path())