from dataclasses import dataclass, asdict
from database.database_book import DatabaseBook
import json
from datetime import datetime


@dataclass
class DatabaseBookMeta:
    id: str
    name: str
    created: str = str(datetime.now())
    last_opened: str = ''

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_meta.json')
        try:
            d = json.load(open(path))
        except FileNotFoundError as e:
            d = {'name': book.book}

        d['id'] = book.book

        return DatabaseBookMeta(**d)

    @staticmethod
    def from_json(book: DatabaseBook, json: dict):
        meta = DatabaseBookMeta.load(book)
        for key, value in json.items():
            setattr(meta, key, value)

        return meta

    def to_json(self):
        return asdict(self)

    def to_file(self, book: DatabaseBook):
        s = json.dumps(self.to_json(), indent=2)
        with open(book.local_path('book_meta.json'), 'w') as f:
            f.write(s)


if __name__ == '__main__':
    b = DatabaseBookMeta.load(DatabaseBook('Graduel'))
    print(b.to_json())
