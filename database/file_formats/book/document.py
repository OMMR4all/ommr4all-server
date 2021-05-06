from datetime import datetime
from typing import List
import uuid


class DocumentConnection:
    def __init__(self, page_id=None, line_id=None):
        self.page_id = page_id
        self.line_id = line_id

    @staticmethod
    def from_json(json: dict):
        return DocumentConnection(
            json.get('page_id', None),
            json.get('line_id', None),

        )

    def to_json(self):
        return {
            "page_id": self.page_id,
            "line_id": self.line_id,
        }

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Document:
    def __init__(self, page_ids, page_names, start: DocumentConnection, end: DocumentConnection,
                 monody_id=None, doc_id=None):
        self.monody_id = monody_id if monody_id else str(uuid.uuid4())
        self.doc_id = doc_id if doc_id else str(uuid.uuid4())
        self.pages_ids: List[int] = page_ids
        self.pages_names: List[str] = page_names
        self.start: DocumentConnection = start
        self.end: DocumentConnection = end

    @staticmethod
    def from_json(json: dict):
        return Document(
            page_ids=json.get('page_ids', []),
            page_names=json.get('pages_names', []),
            monody_id=json.get('monody_id', None),
            doc_id=json.get('doc_id', None),
            start=DocumentConnection.from_json(json.get('start_point', None)),
            end=DocumentConnection.from_json(json.get('end_point', None)),
        )

    def to_json(self):
        return {
            "page_ids": self.pages_ids,
            "pages_names": self.pages_names,
            "monody_id": self.monody_id,
            "doc_id": self.doc_id,
            "start_point": self.start.to_json(),
            "end_point": self.end.to_json(),
        }

    def get_pcgts_of_document(self, book):
        for page in self.pages_names:
            pass


