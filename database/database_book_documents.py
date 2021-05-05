import json
from dataclasses import dataclass, field

from database import DatabasePage
from database.database_book import DatabaseBook
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
from mashumaro import DataClassJSONMixin
from typing import Optional, Dict, List

from database.file_formats.book.document import Document, DocumentConnection
from database.file_formats.book.documents import Documents
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmTypes
from restapi.models.auth import RestAPIUser


class DatabaseBookDocuments():
    def __init__(self, b_id: str = None, monodi_id: int = None, name: str = '', created: datetime = datetime.now(),
                 creator: Optional[RestAPIUser] = None, database_documents: Documents = None):
        self.b_id = b_id
        self.name: str = name
        self.created: datetime = created
        self.database_documents: Documents = database_documents

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

    @staticmethod
    def update_book_documents(book: DatabaseBook):
        d: DatabaseBookDocuments = DatabaseBookDocuments.load(book)
        db_pages: List[DatabasePage] = book.pages()
        document_page_ids = []
        document_page_names = []

        documents: List[Document] = []
        start = None
        for db_page in db_pages:
            page = db_page.pcgts().page
            if start is not None:
                document_page_ids.append(page.p_id)
                document_page_names.append(page.location.page)

            for t_line in page.all_text_lines():

                if t_line.sentence.document_start:
                    if start is None:
                        start = DocumentConnection(line_id=t_line.id, page_id=page.p_id)
                        document_page_ids.append(page.p_id)
                        document_page_names.append(page.location.page)
                    else:
                        documents.append(Document(document_page_ids, document_page_names,
                                                  start=start,
                                                  end=DocumentConnection(line_id=t_line.id, page_id=page.p_id)))
                        document_page_ids = [page.p_id]
                        document_page_names = [page.location.page]

                        start = DocumentConnection(line_id=t_line.id, page_id=page.p_id)
        if start is not None:
            documents.append(Document(document_page_ids, document_page_names,
                                  start=start,
                                  end=DocumentConnection(line_id=None, page_id=None)))
        d.database_documents = Documents(documents=documents)
        return d


if __name__ == '__main__':
    b = DatabaseBookDocuments.update_book_documents(DatabaseBook('demo2'))

    b.to_file(DatabaseBook("demo2"))
    print(b.to_json())
