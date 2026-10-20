"""Export the documents listed in a CM metadata xlsx as Monodi JSON.

Reads the document ids from column A of the active sheet (starting at row 2) and
writes one ``<monody_id>/{data,meta}.json`` per document plus a ``meta.json`` for
the source itself. A ``1`` in the skip column (AQ by default) leaves a row out.

The xlsx is the file produced by ``tools.export.export_docs``.

    python -m tools.export.export_mdocs_specified_by_xlms \
        --book Geesebook2_andreas --source Geesebook2 \
        --xlsx "/tmp/CM Default Metadatendatei.xlsx" --out /tmp/export
"""
import argparse
import json
import os

import openpyxl

from database import DatabaseBook, DatabasePage, DatabaseFile
from database.database_book_documents import DatabaseBookDocuments
from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter

DEFAULT_IIIF_URL = "https://iiif-ls6.informatik.uni-wuerzburg.de/iiif/3/"


def source_meta(id):
    return {
        "id": id,
        "quellensigle": "",
        "herkunftsregion": "",
        "herkunftsort": "",
        "herkunftsinstitution": "",
        "ordenstradition": "",
        "quellentyp": "",
        "bibliotheksort": "",
        "bibliothek": "",
        "bibliothekssignatur": "",
        "kommentar": "",
        "datierung": "",
        "status": "",
        "jahrhundert": "",
        "manifest": "",
        "foliooffset": 0,
        "publish": None
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--book", required=True, help="book the documents belong to")
    parser.add_argument("--source", required=True,
                        help="source id used for the output directory and the IIIF source name")
    parser.add_argument("--xlsx", required=True, help="CM metadata xlsx listing the document ids in column A")
    parser.add_argument("--out", default="/tmp/export", help="output directory (default: %(default)s)")
    parser.add_argument("--skip-column", default="AQ",
                        help="column whose value 1 skips a row (default: %(default)s)")
    parser.add_argument("--editor", default="OMMR4all", help="editor written into the Monodi meta data")
    parser.add_argument("--suffix", default=".jpg", help="image suffix of the source (default: %(default)s)")
    parser.add_argument("--iiif-url", default=DEFAULT_IIIF_URL, help="IIIF base url (default: %(default)s)")
    parser.add_argument("--replace-filename", default="folio_",
                        help="filename prefix handed to the Monodi converter (default: %(default)s)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import django

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    django.setup()

    args = parse_args()
    book = DatabaseBook(args.book)
    documents = DatabaseBookDocuments().load(book)

    sheet_obj = openpyxl.load_workbook(args.xlsx, data_only=True).active

    out_dir = os.path.join(args.out, args.source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(source_meta(args.source), f, indent=4)

    i = 2
    n_exported = 0
    while True:
        doc_id = sheet_obj["A" + str(i)].value
        if not doc_id:
            break
        if sheet_obj[args.skip_column + str(i)].value == 1:
            i += 1
            continue

        document = documents.database_documents.get_document_by_id(doc_id)
        pages = [DatabasePage(book, x) for x in document.pages_names]
        pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]
        try:
            root = PcgtsToMonodiConverter(pcgts, document=document,
                                          replace_filename=args.replace_filename, remove_char="%")
            meta, nodes = root.get_meta_and_notes(
                document=document, editor=args.editor, sourceIIF=args.source,
                doc_source=args.source, suffix=args.suffix, url=args.iiif_url)
        except Exception:
            print(f"row {i}: {[p.local_path() for p in pages]} (document start row {document.start.row})")
            raise

        doc_dir = os.path.join(out_dir, document.monody_id)
        os.makedirs(doc_dir, exist_ok=True)
        with open(os.path.join(doc_dir, "data.json"), "w") as f:
            json.dump(nodes, f, indent=4)
        with open(os.path.join(doc_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        n_exported += 1
        i += 1

    print(f"exported {n_exported} documents to {out_dir}")
