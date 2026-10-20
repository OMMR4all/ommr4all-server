"""Export a ground-truth dataset without the page images.

For every page of the given books that is marked as verified, three files are
written into ``<out>/<book>/<page>/``:

    pcgts.json           the annotations (layout, staff lines, symbols, lyrics)
    page_progress.json   which steps are done/locked for that page
    meta.json            the page meta data

No image is copied, so the result can be published next to a separate image
archive (this is how the GitHub ground-truth releases are built).

    python -m tools.export.export_dataset_no_images --books mul_2_rsync_gt --out /tmp/gt
    python -m tools.export.export_dataset_no_images --books Pa_14819_gt Koeln_Dombibl_1001b_part_gt \
        --out /tmp/gt --name-as Mulhouse2 --all-pages
"""
import argparse
import json
import os

from database import DatabaseBook

DEFAULT_BOOKS = ["mul_2_rsync_gt"]


def to_file(jsonf, filename):
    if not filename.endswith(".json"):
        raise Exception("Invalid file extension of file '{}'".format(filename))
    # first dump to keep the file if an error occurs
    s = json.dumps(jsonf, indent=2)
    with open(filename, 'w') as f:
        f.write(s)


def export_book(book: DatabaseBook, out_dir: str, dir_name: str = None, only_verified: bool = True) -> int:
    book_dir = os.path.join(out_dir, dir_name or book.book)
    n_exported = 0
    for page in book.pages():
        if only_verified and not page.page_progress().verified_allowed():
            continue
        page_dir = os.path.join(book_dir, page.page)
        os.makedirs(page_dir, exist_ok=True)
        # read only: never let the export create a pcgts file in the storage
        pcgts = page.pcgts(create_if_not_existing=False)
        to_file(pcgts.to_json(True), os.path.join(page_dir, "pcgts.json"))
        page.page_progress().to_json_file(os.path.join(page_dir, "page_progress.json"))
        to_file(page.meta().to_json(), os.path.join(page_dir, "meta.json"))
        n_exported += 1
    return n_exported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--books", nargs="+", default=DEFAULT_BOOKS,
                        help="books to export (default: %(default)s)")
    parser.add_argument("--out", default="/tmp", help="output directory (default: %(default)s)")
    parser.add_argument("--name-as", default=None,
                        help="use this directory name instead of the book name; only valid for a single book")
    parser.add_argument("--all-pages", action="store_true",
                        help="export every page, not only the verified ones")
    args = parser.parse_args(argv)
    if args.name_as and len(args.books) > 1:
        parser.error("--name-as can only be used with a single book")
    return args


if __name__ == "__main__":
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()

    args = parse_args()
    for book_name in args.books:
        n = export_book(DatabaseBook(book_name), args.out, args.name_as, not args.all_pages)
        print(f"{book_name}: exported {n} pages to {os.path.join(args.out, args.name_as or book_name)}")
