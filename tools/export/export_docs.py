"""Export the documents of a book as the Corpus Monodicum metadata xlsx.

Consolidates what used to be three near identical copies of this script
(``export/export_docs.py``, ``Rom/export_docs.py`` and
``export/filter_unedited_documents.py``) that differed only in the book name and
in whether flagged documents were skipped.

    # all documents of a book
    python -m tools.export.export_docs --book Rom_1_export

    # skip documents whose text still carries the "$" flag, and write the
    # skipped ones into a second file next to the first one
    python -m tools.export.export_docs --book Graduel_Syn22_03_24 --skip-flagged --write-skipped
"""
import argparse
import os

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments

DEFAULT_FILENAME = 'CM Default Metadatendatei.xlsx'


def write_xls(documents, docs, book, out_dir, filename, editor, flag_char):
    data = documents.database_documents.export_documents_to_xls(
        documents=docs,
        filename=filename,
        editor=editor,
        book=book,
        flag_char=flag_char)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(data)
    print(f"{len(docs)} documents -> {path}")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--book", required=True, help="book to export the documents of")
    parser.add_argument("--out", default="/tmp", help="output directory (default: %(default)s)")
    parser.add_argument("--filename", default=DEFAULT_FILENAME,
                        help="name of the xlsx file (default: %(default)s)")
    parser.add_argument("--editor", default="Ommr4all", help="editor written into the xlsx")
    parser.add_argument("--flag-char", default="$",
                        help="character marking an unfinished passage (default: %(default)s)")
    parser.add_argument("--skip-flagged", action="store_true",
                        help="leave out documents whose text contains the flag character")
    parser.add_argument("--write-skipped", action="store_true",
                        help="additionally write the skipped documents into <filename>_skip.xlsx")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import django

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    django.setup()

    args = parse_args()
    book = DatabaseBook(args.book)
    documents = DatabaseBookDocuments().load(book)
    all_docs = documents.database_documents.documents

    skipped_ids = []
    if args.skip_flagged:
        for doc in all_docs:
            if args.flag_char in doc.get_text_of_document(book):
                skipped_ids.append(doc.monody_id)
                print(f"skipping flagged document {doc.monody_id}")

    docs = [d for d in all_docs if d.monody_id not in skipped_ids]
    write_xls(documents, docs, book, args.out, args.filename, args.editor, args.flag_char)

    if args.write_skipped and skipped_ids:
        skipped = [d for d in all_docs if d.monody_id in skipped_ids]
        base, ext = os.path.splitext(args.filename)
        write_xls(documents, skipped, book, args.out, base + "_skip" + ext, args.editor, args.flag_char)
