from dataclasses import dataclass
from database.database_book import DatabaseBook
import json
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
from mashumaro import DataClassJSONMixin


@dataclass
class DatabaseBookMeta(DataClassJSONMixin):
    id: str
    name: str
    created: str = str(datetime.now())
    last_opened: str = ''
    notationStyle: str = 'french14'

    def default_models_path(self):
        return os.path.join(DEFAULT_MODELS, self.notationStyle)

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_meta.json')
        try:
            d = json.load(open(path))
        except FileNotFoundError as e:
            d = {'name': book.book}

        d['id'] = book.book

        return DatabaseBookMeta.from_dict(d)

    @staticmethod
    def from_book_dict(book: DatabaseBook, json: dict):
        meta = DatabaseBookMeta.load(book)
        for key, value in json.items():
            setattr(meta, key, value)

        return meta

    def to_file(self, book: DatabaseBook):
        s = self.to_json(indent=2)
        with open(book.local_path('book_meta.json'), 'w') as f:
            f.write(s)


if __name__ == '__main__':
    b = DatabaseBookMeta.load(DatabaseBook('Graduel'))
    print(b.to_json())
