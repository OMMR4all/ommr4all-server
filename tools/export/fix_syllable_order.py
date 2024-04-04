from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.pcgts.page import Sentence

if __name__ == "__main__":

    book = DatabaseBook('Graduel_Syn22_03_24')

    pages = book.pages()
    for i in pages:
        page = i.pcgts().page
        annotation = page.annotations
        change = False
        for t in page.all_text_lines():
            syls = []
            sentence: Sentence = t.sentence
            syllabels = sentence.syllables
            skip = False
            for syl in syllabels:

                con = annotation.get_connection_of_syllable(syl)
                if con:
                    syls.append((syl, con.note.coord.x))
                else:
                    skip = True
                    break
            if skip:
                continue
            text = t.text()

            syls.sort(key=lambda x: x[1])
            t.sentence = Sentence([s for s, _ in syls])
            if text != t.sentence.text():
                print(i.page)
                change = True
                print(sentence.text())
                print(t.sentence.text())
                print("ERROR")
        i.pcgts().to_file(i.file('pcgts').local_path())


