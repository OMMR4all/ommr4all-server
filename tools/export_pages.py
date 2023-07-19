import os
import shutil

from database import DatabaseBook
path = "/tmp/"
if __name__ == "__main__":
    book = DatabaseBook("mul_2_rsync_gt")
    pages = book.page_names()
    pages_ = book.pages()

    if not os.path.exists(os.path.join(path, book.book)):
        os.mkdir(os.path.join(path, book.book))

    for i in pages_:
        shutil.copyfile(i.file("color_original").local_path(), os.path.join(path, book.book, i.page+".jpg", ) )
        pass