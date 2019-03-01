from dataclasses import dataclass, asdict
from main.book import Book
import json
from typing import List
from datetime import datetime
import os


@dataclass
class BookMeta:
    id: str
    name: str
    created: str = str(datetime.now())
    last_opened: str = ''

    @staticmethod
    def load(book: Book):
        path = book.local_path('book_meta.json')
        try:
            d = json.load(open(path))
        except FileNotFoundError as e:
            d = {'name': book.book}

        d['id'] = book.book

        return BookMeta(**d)

    @staticmethod
    def from_json(book: Book, json: dict):
        meta = BookMeta.load(book)
        for key, value in json.items():
            setattr(meta, key, value)

        return meta

    def to_json(self):
        return asdict(self)

    def to_file(self, book: Book):
        s = json.dumps(self.to_json(), indent=2)
        with open(book.local_path('book_meta.json'), 'w') as f:
            f.write(s)


if __name__ == '__main__':
    b = BookMeta.load(Book('Graduel'))
    print(b.to_json())
