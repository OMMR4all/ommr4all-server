import itertools
import os

from database import DatabaseBook

if __name__ == "__main__":

    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings

    # get all books from database
    pages = DatabaseBook("mul_2_gt_22_03").pages()
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files
    from xlwt import Workbook

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    ind = 0
    for page in pages:
        page_id = page.page
        lines = page.pcgts().page.all_text_lines()
        music_lines = page.pcgts().page.all_music_lines()
        symbols = list(itertools.chain.from_iterable([t.symbols for t in music_lines]))

        if len(music_lines) == 0:
            continue
        symbols_per_line = len(symbols) / len(music_lines)
        for ind2, i in enumerate(lines):
            if "$" in i.text():
                sheet1.write(ind, 0, str(i.text()))
                sheet1.write(ind, 1, str(len(symbols)))
                sheet1.write(ind, 2, str(symbols_per_line))
                sheet1.write(ind, 3, str(page_id))
                ind += 1
    wb.save("/tmp/eval_data.xls")


