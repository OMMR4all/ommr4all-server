import os

from database import DatabaseBook
from database.file_formats.performance.pageprogress import Locks

if __name__ == '__main__':
    import django
    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all books from database
    book = DatabaseBook("Winterburger")
    pages = book.pages()
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files
    excepted_ids = []
    for page in pages:
        page_id = page.page
        annotation = page.pcgts().page.annotations
        c_progress = page.page_progress()
        c_progress.locked[Locks.STAFF_LINES] = True
        page.set_page_progress(c_progress)
        page.save_page_progress()
