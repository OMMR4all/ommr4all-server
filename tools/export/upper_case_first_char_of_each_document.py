from database import DatabaseBook

if __name__ == "__main__":

    book = DatabaseBook('mul_2_gt_22_03')

    for i in book.pages():
        for t in i.pcgts().page.all_text_lines():
            if t.document_start:
                if len(t.text()) > 0 and len(t.sentence.syllables) > 0:
                    t.sentence.syllables[0].text = (t.sentence.syllables[0].text[0].upper() +
                                                    t.sentence.syllables[0].text[1:])
        i.pcgts().to_file(i.file('pcgts').local_path())
