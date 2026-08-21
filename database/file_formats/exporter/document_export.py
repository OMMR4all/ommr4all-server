"""Exporting a single document (chant) of a book.

The one place that knows how to turn a Document into a downloadable file, shared by the
per-document download endpoint and the bulk documents_export task runner so the two cannot
drift apart. Everything here is document-scoped: the page-wide exports live in
restapi/views/bookaccess.py (BookDownloaderView) and are a different feature.
"""
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from database import DatabaseBook, DatabaseFile, DatabasePage
from database.file_formats import PcGts
from database.file_formats.book.document import Document


def document_file_name(doc: Document) -> str:
    """Stable, filesystem-safe name for a document, used for zip entries and downloads."""
    initium = doc.document_meta_infos.initium if doc.document_meta_infos and doc.document_meta_infos.initium \
        else doc.textinitium
    initium = (initium or '').replace('-', '')
    initium = re.sub(r'[^\w]+', '_', initium).strip('_')[:60]
    return '_'.join(filter(None, [initium, doc.doc_id]))


def pcgts_of_document(book: DatabaseBook, doc: Document) -> List[PcGts]:
    pages = [DatabasePage(book, name) for name in doc.pages_names]
    return [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in pages]


def document_line_ids_by_page(book: DatabaseBook, doc: Document) -> Dict[str, List[str]]:
    """The ids of the text lines belonging to the document, grouped by page name.

    Ids rather than Line objects on purpose: the caller usually holds its own PcGts instances,
    for which the Line objects returned here would not be identical.
    """
    by_page: Dict[str, List[str]] = OrderedDict()
    for line, page in doc.get_page_line_of_document(book):
        by_page.setdefault(page.page, []).append(line.id)
    return by_page


def monodi_json_of_document(book: DatabaseBook, doc: Document, editor: str) -> dict:
    """The Monodi+ document file: {"document": <metadata>, "notes": <content>}."""
    from database.file_formats.exporter.monodi.monodi2_exporter import (
        MonodiExportSettings, PcgtsToMonodiConverter,
    )
    converter = PcgtsToMonodiConverter(pcgts_of_document(book, doc), document=doc)
    # source id and iiif urls come from the book meta, not from a hard coded manuscript
    return converter.get_Monodi_json(document=doc, editor=editor,
                                     settings=MonodiExportSettings.from_book(book))


def monodi_notes_of_document(book: DatabaseBook, doc: Document) -> dict:
    """Just the note container of the document, without the metadata envelope.

    The shape the page-wide export in BookDownloaderView produces, for workflows that consume
    the notes alone; monodi_json_of_document is the complete Monodi+ document file.
    """
    from database.file_formats.exporter.monodi.monodi2_exporter import PcgtsToMonodiConverter
    return PcgtsToMonodiConverter(pcgts_of_document(book, doc), document=doc).root.to_json()


def mei_files_of_document(book: DatabaseBook, doc: Document) -> List[Tuple[str, str]]:
    """MEI4 of the document, one entry (page_name, xml) per page it actually covers.

    One file per page rather than one merged file: the converter emits a <score> per page into
    a single <mdiv>, which the MEI schema does not accept.
    """
    from database.file_formats.exporter.mei.pcgts_to_mei4_exporter import PcgtsToMeiConverter
    line_ids = document_line_ids_by_page(book, doc)
    files = []
    for pcgts in pcgts_of_document(book, doc):
        page_name = pcgts.page.location.page
        converter = PcgtsToMeiConverter(pcgts, document_line_ids=line_ids.get(page_name, []))
        if not converter.has_content:
            continue
        files.append((page_name, converter.to_string()))
    return files
