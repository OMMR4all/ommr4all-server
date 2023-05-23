import json
import time
from collections import namedtuple
from dataclasses import dataclass, field

from database import DatabasePage
from database.database_book import DatabaseBook
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
from typing import Optional, Dict, List

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


class DatabaseBookDocuments:
    def __init__(self, b_id: str = None, monodi_id: int = None, name: str = '', created: datetime = datetime.now(),
                 creator: Optional[RestAPIUser] = None, database_documents: Documents = None):
        self.b_id = b_id
        self.name: str = name
        self.created: datetime = created
        self.database_documents: Documents = database_documents

    def __iter__(self):
        return iter(self.database_documents.documents)

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_documents.json')
        try:
            with open(path) as f:
                d = DatabaseBookDocuments.from_book_json(book, json.load(f))
        except FileNotFoundError:
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

    @staticmethod
    def from_json(json: dict):
        return DatabaseBookDocuments(
            name=json.get('name', ""),
            created=datetime.fromisoformat(json.get('created', datetime.now().isoformat())),
            database_documents=Documents.from_json(json.get('database_documents', []))
        )

    def to_json(self):
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "database_documents": self.database_documents.to_json() if self.database_documents else []
        }

    def get_documents_of_page(self, page: Page, only_start=False) -> List[DocSpanType]:

        docs: List[DocSpanType] = []
        for ind, i in enumerate(self.database_documents.documents):
            if i.start.page_id == page.p_id:
                docs.append(DocSpanType(p_start=i.start.page_id, p_end=i.end.page_id, doc=i, index=ind))
            if only_start and i.end.page_id == page.p_id:
                docs.append(DocSpanType(p_start=i.start.page_id, p_end=i.end.page_id, doc=i, index=ind))

        return docs

    def update_documents_of_page(self, page: DatabasePage, book: DatabaseBook):
        def get_current_documents_of_page(page: Page):
            doc_starts = namedtuple("DocStart", "line prev_line index")
            line: List[doc_starts] = []
            prev_line = None
            for ind, t_line in enumerate(page.reading_order.reading_order):
                if t_line.document_start:
                    line.append(doc_starts(t_line, prev_line, ind))
                prev_line = t_line
            return line

        page_p = page.pcgts().page

        docs = self.get_documents_of_page(page_p)

        pages = book.pages(load_pcgts=True)

        index = [index for index, i in enumerate(pages) if i.pcgts().page.p_id == page_p.p_id][0]
        prev_page = pages[index - 1].pcgts().page
        if index > 0:
            prev_docs = self.get_documents_of_page(prev_page)
            prev_docs.sort(key=lambda i: i.index)
            prev_doc = prev_docs[-1].doc if len(prev_docs) > 0 else None
        else:
            prev_doc = None
        # prev_doc = [i for i in docs if i.p_start != page_p.p_id]
        end_doc = [i for i in docs if i.p_end != page_p.p_id]
        # db_pages: List[DatabasePage] = book.pages()
        starts = get_current_documents_of_page(page_p)
        new_docs = []

        for ind, line in enumerate(starts):

            if prev_doc:
                if line.prev_line is not None:
                    prev_doc.end = DocumentConnection(line_id=line.prev_line.id, page_id=page_p.p_id,
                                                      row=line.index - 1,
                                                      page_name=page_p.location.page)
                    prev_doc.textline_count = 0
                    prev_doc.update_textline_count(book=book)
                else:
                    all_text_lines = prev_page.get_reading_order()
                    lline = all_text_lines[-1]
                    prev_doc.end = DocumentConnection(line_id=lline, page_id=prev_page.p_id,
                                                      row=len(all_text_lines),
                                                      page_name=prev_page.location.page)
                    prev_doc.update_textline_count(book=book)

            if ind + 1 == len(starts):
                if len(end_doc) > 0:
                    end_doc[0].doc.start = DocumentConnection(line_id=line.line.id, page_id=page_p.p_id, row=line.index,
                                                              page_name=page_p.location.page)
                    end_doc[0].doc.update_textline_count(book=book)
                    end_doc[0].doc.textinitium = line.line.sentence.text(True)

                    continue
                else:
                    search = True
                    i = 1
                    pages_id = [page_p.p_id]
                    page_location = [page_p.location.page]
                    prev_page = page_p
                    prev_line = line

                    while search:
                        if index + i < len(pages):
                            next_page = pages[index + 1].pcgts().page
                            pages_id.append(next_page.p_id)
                            page_location.append(next_page.location.page)

                            for ind, li in enumerate(next_page.get_reading_order()):
                                if li.document_start:
                                    if i == 1 and ind == 0:
                                        pages_id = pages_id[:-1]
                                        page_location = page_location[:-1]

                                    new_docs.append(Document([pages_id], [page_location],
                                                             start=DocumentConnection(line_id=line.line.id,
                                                                                      page_id=page_p.p_id,
                                                                                      row=line.index,
                                                                                      page_name=page_p.location.page),
                                                             end=DocumentConnection(line_id=prev_line.line.id,
                                                                                    page_id=prev_page.p_id,
                                                                                    row=prev_line.index,
                                                                                    page_name=prev_page.location.page),
                                                             textinitium=line.line.sentence.text(True),
                                                             textline_count=0)
                                                    )
                                prev_line = li
                                prev_page = next_page
                                search = False
                                break
                        else:
                            new_docs.append(Document([pages_id], [page_location],
                                                     start=DocumentConnection(line_id=line.line.id,
                                                                              page_id=page_p.p_id,
                                                                              row=line.index,
                                                                              page_name=page_p.location.page),
                                                     end=DocumentConnection(line_id=prev_line.id,
                                                                            page_id=prev_page.p_id,
                                                                            row=prev_line.index,
                                                                            page_name=prev_page.location.page),
                                                     textinitium=line.line.sentence.text(True).replace("-", "").replace("~", ""),
                                                     textline_count=0)
                                            )
                            search = False
                            break
                    continue
            next_line = starts[ind + 1]
            new_docs.append(Document([page_p.p_id], [page_p.location.page],
                                     start=DocumentConnection(line_id=line.line.id, page_id=page_p.p_id, row=line.index,
                                                              page_name=page_p.location.page),
                                     end=DocumentConnection(line_id=next_line.prev_line.id, page_id=page_p.p_id,
                                                            row=next_line.index -1,
                                                            page_name=page_p.location.page),
                                     textinitium=line.line.sentence.text(True).replace("-", "").replace("~", ""), textline_count=0)
                            )
        for doc in new_docs:
            doc.update_textline_count(book)

        for line in sorted(docs, key=lambda x: x.index, reverse= True):
            del self.database_documents.documents[line.index]
        self.database_documents.documents += new_docs

        return self

    @staticmethod
    def update_book_documents(book: DatabaseBook):
        d: DatabaseBookDocuments = DatabaseBookDocuments.load(book)
        db_pages: List[DatabasePage] = book.pages(True)
        document_page_ids = []
        document_page_names = []
        textinitium = ''
        documents: List[Document] = []
        start = None
        line_count = 0
        for page_ind, db_page in enumerate(db_pages):
            page = db_page.pcgts().page
            if start is not None:
                document_page_ids.append(page.p_id)
                document_page_names.append(page.location.page)

            for ind, t_line in enumerate(page.reading_order.reading_order, start=1):

                if t_line.document_start:
                    if start is None:
                        start = DocumentConnection(line_id=t_line.id, page_id=page.p_id, row=ind,
                                                   page_name=page.location.page)
                        textinitium = t_line.sentence.text(True)
                        document_page_ids.append(page.p_id)
                        document_page_names.append(page.location.page)
                    else:
                        end_row = ind - 1 if ind - 1 != 0 else len(db_pages[page_ind - 1].pcgts().page.all_text_lines())
                        end_page = page.location.page if ind - 1 != 0 else db_pages[
                            page_ind - 1].pcgts().page.location.page
                        documents.append(Document(document_page_ids, document_page_names,
                                                  start=start,
                                                  end=DocumentConnection(line_id=t_line.id, page_id=page.p_id,
                                                                         row=end_row, page_name=end_page),
                                                  textinitium=textinitium, textline_count=line_count))
                        document_page_ids = [page.p_id]
                        document_page_names = [page.location.page]

                        start = DocumentConnection(line_id=t_line.id, page_id=page.p_id, row=ind,
                                                   page_name=page.location.page)
                        textinitium = t_line.sentence.text(True)
                        line_count = 0
                if start is not None:
                    line_count += 1

        if start is not None:
            db_page = db_pages[-1]
            page = db_page.pcgts().page
            lines = page.all_text_lines()

            documents.append(Document(document_page_ids, document_page_names,
                                      start=start,
                                      end=DocumentConnection(line_id=lines[-1].id if len(lines) > 0 else None,
                                                             page_id=page.p_id, row=len(lines),
                                                             page_name=page.location.page), textinitium=textinitium,
                                      textline_count=line_count))

        updated_documents: List[Document] = []
        for doc in documents:
            if d.database_documents:
                found = False
                for orig_doc in d.database_documents.documents:
                    if doc.start == orig_doc.start:
                        updated_doc = orig_doc
                        updated_doc.pages_names = doc.pages_names
                        updated_doc.pages_ids = doc.pages_ids
                        updated_doc.end = doc.end
                        updated_doc.textinitium = doc.textinitium
                        updated_doc.textline_count = doc.textline_count
                        updated_documents.append(updated_doc)
                        found = True
                        break
                if not found:
                    updated_documents.append(doc)


            else:
                updated_documents.append(doc)

        d.database_documents = Documents(documents=updated_documents)

        return d


if __name__ == '__main__':
    b = DatabaseBookDocuments.update_book_documents(DatabaseBook('demo2'))

    b.to_file(DatabaseBook("demo2"))
    print(b.to_json())
