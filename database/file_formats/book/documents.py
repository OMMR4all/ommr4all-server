import json
from datetime import datetime
from typing import List

from database.file_formats.book.document import Document


class Documents:
    def __init__(self, documents: List[Document]):
        self.documents: List[Document] = documents

    @staticmethod
    def from_json(json: dict):
        if type(json) == dict:
            return Documents(
                documents=[Document.from_json(l) for l in json.get('documents', [])]
            )
        else:
            return None

    def to_json(self):
        return {
            "documents": [document.to_json() for document in self.documents] if self.documents else [],
        }

    def get_document_by_id(self, id):
        for x in self.documents:
            if x.doc_id == id:
                return x
        return None


if __name__ == '__main__':
    documents = Documents([Document("1", ["page1", "page2"], 1000), Document("2", ["page2", "page3"], 1002)])
    js = json.dumps(documents.to_json(), indent=2)
    documents = Documents.from_json(json.loads(js))
    print(documents.to_json())
    # print(json.dumps(s.to_json(), indent=2))
