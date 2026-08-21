import json
import time
from dataclasses import dataclass, field

from filelock import FileLock

from database import DatabasePage
from database.database_book import DatabaseBook
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from database.file_formats.book.document import Document, DocumentConnection
from database.file_formats.book.documents import Documents
from database.file_formats.pcgts import Line, Page
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmTypes
from restapi.models.auth import RestAPIUser


@dataclass
class DocSpanType:
    p_start: str
    p_end: str
    doc: Document
    index: int


@dataclass
class FragmentLine:
    """A text line of a page's reading order, reduced to what document assembly needs."""
    id: str
    start: bool = False
    text: str = ''

    @staticmethod
    def from_json(d: dict) -> 'FragmentLine':
        return FragmentLine(id=d.get('id'), start=d.get('start', False), text=d.get('text', ''))

    def to_json(self) -> dict:
        d = {'id': self.id}
        if self.start:
            d['start'] = True
            d['text'] = self.text
        return d


@dataclass
class PageDocumentFragment:
    """Per-page slice of the pcgts needed to assemble the documents of a book.

    Cached in book_documents.json keyed by the pcgts file's mtime, so a document
    update only has to reparse the pages that actually changed.
    """
    page_name: str
    p_id: str
    mtime: float
    lines: List[FragmentLine] = field(default_factory=list)

    @staticmethod
    def extract(db_page: DatabasePage, mtime: float) -> 'PageDocumentFragment':
        page = db_page.pcgts_cached().page
        return PageDocumentFragment(
            page_name=db_page.page,
            p_id=page.p_id,
            mtime=mtime,
            # the reading order (lyric lines only) is the one ordering documents are indexed in;
            # page.all_text_lines() is a different, block-ordered list and must not be mixed in
            lines=[FragmentLine(id=t_line.id, start=t_line.document_start,
                                text=t_line.sentence.text(True) if t_line.document_start else '')
                   for t_line in page.reading_order.reading_order],
        )

    @staticmethod
    def from_json(d: dict) -> 'PageDocumentFragment':
        return PageDocumentFragment(
            page_name=d.get('page_name'),
            p_id=d.get('p_id'),
            mtime=d.get('mtime', 0.0),
            lines=[FragmentLine.from_json(line) for line in d.get('lines', [])],
        )

    def to_json(self) -> dict:
        return {
            'page_name': self.page_name,
            'p_id': self.p_id,
            'mtime': self.mtime,
            'lines': [line.to_json() for line in self.lines],
        }


class DatabaseBookDocuments:
    # Bump whenever _assemble_documents changes what it produces for unchanged pages: stored
    # documents carrying an older version are reassembled once instead of being served as is.
    DOCUMENTS_FORMAT_VERSION = 1

    def __init__(self, b_id: str = None, monodi_id: int = None, name: str = '', created: datetime = datetime.now(),
                 creator: Optional[RestAPIUser] = None, database_documents: Documents = None,
                 page_fragments: Optional[List[PageDocumentFragment]] = None, version: int = 0):
        self.b_id = b_id
        self.name: str = name
        self.created: datetime = created
        self.database_documents: Documents = database_documents
        # Per-page snapshots (keyed by pcgts mtime) from which the documents were assembled.
        # Only pages whose pcgts file changed since then need to be reparsed.
        self.page_fragments: Optional[List[PageDocumentFragment]] = page_fragments
        self.version: int = version

    @staticmethod
    def lock(book: DatabaseBook) -> FileLock:
        """Inter-process lock guarding read-modify-write cycles on book_documents.json.

        Not reentrant across FileLock instances: don't call update_book_documents_cached
        (which acquires it internally) while holding it.
        """
        return FileLock(book.local_path('book_documents.json.lock'), timeout=30)

    def __iter__(self):
        return iter(self.database_documents.documents)

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_documents.json')
        try:
            with open(path) as f:
                d = DatabaseBookDocuments.from_book_json(book, json.load(f))
        except FileNotFoundError as e:
            d = DatabaseBookDocuments(b_id=book.book)

        return d

    @staticmethod
    def from_book_json(book: DatabaseBook, json: dict):
        documents = DatabaseBookDocuments.from_json(json)
        documents.b_id = book.book
        if len(documents.name) == 0:
            documents.name = book.book
        return documents

    def to_file(self, book: DatabaseBook):
        self.b_id = book.book
        s = self.to_json()
        with open(book.local_path('book_documents.json'), 'w') as f:
            js = json.dumps(s, indent=2)
            f.write(js)
        from database.book_index import safe_index_documents
        safe_index_documents(book)

    @staticmethod
    def from_json(json: dict):
        page_fragments = json.get('page_fragments', None)
        return DatabaseBookDocuments(
            name=json.get('name', ""),
            created=datetime.fromisoformat(json.get('created', datetime.now().isoformat())),
            database_documents=Documents.from_json(json.get('database_documents', [])),
            # Files written before the fragment cache (only a coarse 'pcgts_state') yield None,
            # which forces a full re-extraction on the next update.
            page_fragments=[PageDocumentFragment.from_json(f) for f in page_fragments]
            if page_fragments is not None else None,
            # absent in files written before the version field existed
            version=json.get('version', 0),
        )

    def to_json(self):
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "version": self.version,
            "database_documents": self.database_documents.to_json() if self.database_documents else [],
            "page_fragments": [f.to_json() for f in self.page_fragments]
            if self.page_fragments is not None else None,
        }

    def get_documents_of_page(self, page: Page, only_start=False) -> List[DocSpanType]:

        docs: List[DocSpanType] = []
        for ind, i in enumerate(self.database_documents.documents):
            if i.start.page_id == page.p_id:
                docs.append(DocSpanType(p_start=i.start.page_id, p_end=i.end.page_id, doc=i, index=ind))
            if only_start and i.end.page_id == page.p_id:
                docs.append(DocSpanType(p_start=i.start.page_id, p_end=i.end.page_id, doc=i, index=ind))

        return docs

    @staticmethod
    def _update_page_fragments(book: DatabaseBook, fragments: Optional[List[PageDocumentFragment]]) \
            -> Tuple[List[PageDocumentFragment], bool]:
        """Revalidate the per-page fragments against the pcgts files on disk.

        Returns the up-to-date fragments (in page order) and whether anything changed,
        i.e. whether the documents need to be reassembled. Only pages whose pcgts mtime
        differs from the cached fragment are reparsed.
        """
        cached: Dict[str, PageDocumentFragment] = {f.page_name: f for f in (fragments or [])}
        updated: List[PageDocumentFragment] = []
        changed = fragments is None
        for db_page in book.pages():
            path = db_page.file('pcgts').local_path()
            if not os.path.exists(path):
                db_page.pcgts()  # creates the missing pcgts file
            mtime = os.path.getmtime(path)
            fragment = cached.get(db_page.page)
            if fragment is None or fragment.mtime != mtime:
                fragment = PageDocumentFragment.extract(db_page, mtime)
                changed = True
            updated.append(fragment)
        if len(updated) != len(cached):
            changed = True
        return updated, changed

    @staticmethod
    def _assemble_documents(fragments: List[PageDocumentFragment]) -> List[Document]:
        """Assemble the documents (chants) of the whole book from the per-page fragments.

        A line flagged as document start opens a chant and closes the previous one.

        Everything is derived from the lines actually taken: a page enters pages_names only
        once one of its lines belongs to the document, and end.page_name/end.row describe the
        last line taken. See DocumentConnection for the exclusive-cursor convention of end.
        """
        documents: List[Document] = []
        document_page_ids: List[str] = []
        document_page_names: List[str] = []
        textinitium = ''
        start: Optional[DocumentConnection] = None
        line_count = 0
        # last line actually taken by the open document
        last_row = 0
        last_fragment: Optional[PageDocumentFragment] = None

        for fragment in fragments:
            for ind, line in enumerate(fragment.lines, start=1):
                if line.start:
                    if start is not None:
                        documents.append(Document(document_page_ids, document_page_names,
                                                  start=start,
                                                  end=DocumentConnection(line_id=line.id, page_id=fragment.p_id,
                                                                         row=last_row,
                                                                         page_name=last_fragment.page_name),
                                                  textinitium=textinitium, textline_count=line_count))
                        document_page_ids = []
                        document_page_names = []
                        line_count = 0
                    start = DocumentConnection(line_id=line.id, page_id=fragment.p_id, row=ind,
                                               page_name=fragment.page_name)
                    textinitium = line.text
                if start is not None:
                    # compare page names, not p_ids: a page that was never saved mints a new
                    # p_id on every load, so two pages can transiently share one
                    if not document_page_names or document_page_names[-1] != fragment.page_name:
                        document_page_ids.append(fragment.p_id)
                        document_page_names.append(fragment.page_name)
                    line_count += 1
                    last_row, last_fragment = ind, fragment

        if start is not None:
            documents.append(Document(document_page_ids, document_page_names,
                                      start=start,
                                      # '' never equals a real line id: the last document of the
                                      # book has no successor to stop at and runs to the end
                                      end=DocumentConnection(line_id='', page_id=last_fragment.p_id,
                                                             row=last_row,
                                                             page_name=last_fragment.page_name),
                                      textinitium=textinitium, textline_count=line_count))
        return documents

    def _merge_into_existing(self, documents: List[Document]) -> List[Document]:
        """Adopt freshly assembled documents, keeping doc_id/monody_id/meta infos of
        existing documents whose start connection is unchanged."""
        if not self.database_documents:
            return documents
        updated_documents: List[Document] = []
        for doc in documents:
            for orig_doc in self.database_documents.documents:
                if doc.start == orig_doc.start:
                    orig_doc.pages_names = doc.pages_names
                    orig_doc.pages_ids = doc.pages_ids
                    orig_doc.end = doc.end
                    orig_doc.textinitium = doc.textinitium
                    orig_doc.textline_count = doc.textline_count
                    updated_documents.append(orig_doc)
                    break
            else:
                updated_documents.append(doc)
        return updated_documents

    @staticmethod
    def update_book_documents_cached(book: DatabaseBook) -> 'DatabaseBookDocuments':
        """Bring the documents of the book up to date, reparsing only the pages whose
        pcgts file changed since the last computation.

        Persists the result if anything changed. Guarded by the book_documents file lock,
        so concurrent requests cannot clobber each other.
        """
        with DatabaseBookDocuments.lock(book):
            d = DatabaseBookDocuments.load(book)
            fragments, changed = DatabaseBookDocuments._update_page_fragments(book, d.page_fragments)
            if not changed and d.database_documents is not None \
                    and d.version == DatabaseBookDocuments.DOCUMENTS_FORMAT_VERSION:
                return d
            d.database_documents = Documents(documents=d._merge_into_existing(
                DatabaseBookDocuments._assemble_documents(fragments)))
            d.page_fragments = fragments
            d.version = DatabaseBookDocuments.DOCUMENTS_FORMAT_VERSION
            d.to_file(book)
            return d

    @staticmethod
    def update_book_documents(book: DatabaseBook) -> 'DatabaseBookDocuments':
        """Full recomputation ignoring the fragment cache. Does not persist."""
        d: DatabaseBookDocuments = DatabaseBookDocuments.load(book)
        fragments, _ = DatabaseBookDocuments._update_page_fragments(book, None)
        d.database_documents = Documents(documents=d._merge_into_existing(
            DatabaseBookDocuments._assemble_documents(fragments)))
        d.page_fragments = fragments
        d.version = DatabaseBookDocuments.DOCUMENTS_FORMAT_VERSION
        return d


if __name__ == '__main__':
    b = DatabaseBookDocuments.update_book_documents(DatabaseBook('demo2'))

    b.to_file(DatabaseBook("demo2"))
    print(b.to_json())
