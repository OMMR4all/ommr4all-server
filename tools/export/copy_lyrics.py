from database import DatabaseBook, DatabasePage
from database.file_formats.pcgts.page import Sentence

if __name__ == "__main__":
    book_gt = DatabaseBook('Geesebook1gt')
    book_pred = DatabaseBook('Geesebook1_base_predict_gt_text')

    pages_gt = book_gt.pages()
    pages_pred = book_pred.pages()

    for page_gt, page_pred in zip(pages_gt, pages_pred):
        page_gt: DatabasePage = page_gt
        page_pred: DatabasePage = page_pred
        print(f"GT:{page_gt.pcgts().page.location.page} pred: {page_pred.pcgts().page.location.page}")
        if page_gt.pcgts().page.location.page == page_pred.pcgts().page.location.page:
            lines_gt = page_gt.pcgts().page.all_text_lines(only_lyric=True)
            lines_pred = page_pred.pcgts().page.all_text_lines(only_lyric=True)
            for l_gt, l_pred in zip(lines_gt, lines_pred):
                print(l_gt.sentence.text())
                print(l_pred.sentence.text())
                print()
                sentence = Sentence.from_string(l_gt.sentence.text())
                l_pred.sentence = sentence
            print("same")
        else:
            print("different")
        page_pred.pcgts().to_file(page_pred.file('pcgts').local_path())

        print(page_gt)
        print(page_pred)
        print()