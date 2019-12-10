from argparse import ArgumentParser
from prettytable import PrettyTable
from database import DatabaseBook
from database.tools.book_statistics import compute_book_statistics

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--ignore-page", nargs="+", default=[])

    args = parser.parse_args()

    all_book_counts = [compute_book_statistics(DatabaseBook(book), args.ignore_page) for book in args.books]

    pt = PrettyTable([n for n, _ in all_book_counts[0].to_dict().items()])
    for book_counts in all_book_counts:
        pt.add_row([v for _, v in book_counts.to_dict().items()])

    print(pt)


