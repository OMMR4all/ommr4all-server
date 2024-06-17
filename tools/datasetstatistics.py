from argparse import ArgumentParser
from typing import List

from prettytable import PrettyTable
from database import DatabaseBook, DatabasePage
from database.tools.book_statistics import compute_book_statistics

#--books mul_2_rsync_gt --specific-pages-only folio_0256 folio_0256v folio_0257 folio_0257v folio_0258 folio_0258v folio_0259 folio_0259v folio_0260 folio_0260v folio_0261 folio_0261v folio_0262 folio_0262v folio_0263 folio_0263v folio_0264 folio_0264v folio_0265 folio_0265v
#--books mul_2_23_gt_ina

def gen_ignore_pages(pages: List[str], books: List[DatabaseBook]):
    print("123")
    print(page.page for page in books[0].pages())
    for i in books[0].pages():
        print(i.page)
    pages = sum([[page.page for page in book.pages() if not any([s in page.page for s in pages])] for book in books],
                [])
    return pages


def convert_to_int(st):
    s = ''.join(i for i in st if i.isdigit())
    while len(s) > 0 and s[0] == "0":
        s = s[1:]
    print(int(s))
    return int(s)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--ignore-page", nargs="+", default=[])
    parser.add_argument("--specific-pages-only", nargs="+", default=[])
    parser.add_argument("--below-specific-page", type=int)
    args = parser.parse_args()
    if len(args.specific_pages_only) > 0:
        ignore_pages = gen_ignore_pages(args.specific_pages_only, [DatabaseBook(book) for book in args.books])
        all_book_counts = [compute_book_statistics(DatabaseBook(book), ignore_pages) for book in args.books]
    elif args.below_specific_page:
        books = [DatabaseBook(book) for book in args.books]
        pages = [page.page for book in books for page in book.pages() if convert_to_int(page.page) < args.below_specific_page]
        ignore_pages = gen_ignore_pages(pages, [DatabaseBook(book) for book in args.books])
        all_book_counts = [compute_book_statistics(DatabaseBook(book), ignore_pages) for book in args.books]
    else:
        all_book_counts = [compute_book_statistics(DatabaseBook(book), args.ignore_page) for book in args.books]

    pt = PrettyTable([n for n, _ in all_book_counts[0].to_dict().items()])
    for book_counts in all_book_counts:
        pt.add_row([v for _, v in book_counts.to_dict().items()])

    print(pt)

