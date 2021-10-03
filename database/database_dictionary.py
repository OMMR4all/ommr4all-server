import json
import os
from datetime import datetime
from typing import List

from dataclasses import dataclass

from dataclasses_json import dataclass_json

from database import DatabaseBook
from ommr4all import settings


@dataclass_json
@dataclass
class WordFrequency:
    word: str
    frequency: int


@dataclass_json
@dataclass
class WordFrequencyDict:
    freq_list: List[WordFrequency]

    #def get_n_most_similar_words(self, word, n=5):
    #    probs = {}
     #   total = sum([x.frequency for x in self.freq_list])
     #   for k in self.freq_list:
     #       probs[k.word] = k.frequency / total
     #   if word in probs:
     #       return "yay"
     #   else:
     #       similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq_dict.keys()]




class DatabaseDictionary:
    def __init__(self, b_id: str = None, name: str = '', created: datetime = datetime.now(),
                 dictionary=None
                 ):
        if dictionary is None:
            dictionary = []
        self.b_id = b_id
        self.name: str = name
        self.created: datetime = created
        self.dictionary: WordFrequencyDict = dictionary

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_dictionary.json')
        try:
            with open(path) as f:
                d = DatabaseDictionary.from_book_json(book, json.load(f))
        except FileNotFoundError:
            try:
                path = os.path.join(settings.BASE_DIR,
                                    'internal_storage',
                                    'default_dictionary',
                                    'default_dictionary.json')
                with open(path) as f:

                    d = DatabaseDictionary.from_book_json(book, json.load(f))
            except FileNotFoundError:
                d = DatabaseDictionary(b_id=book.book)

        return d

    @staticmethod
    def from_book_json(book: DatabaseBook, json: dict):
        dictionary = DatabaseDictionary.from_json(json)
        dictionary.b_id = book.book
        if len(dictionary.name) == 0:
            dictionary.name = book.book
        return dictionary

    def to_file(self, book: DatabaseBook):
        self.b_id = book.book
        s = self.to_json()
        with open(book.local_path('book_dictionary.json'), 'w') as f:
            js = json.dumps(s, indent=2)
            f.write(js)

    @staticmethod
    def from_json(json: dict):
        return DatabaseDictionary(
            name=json.get('name', ""),
            created=datetime.fromisoformat(json.get('created', datetime.now().isoformat())),
            dictionary=WordFrequencyDict.from_dict(json.get('dictionary'))

        )

    def to_json(self):
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "dictionary": self.dictionary.to_dict() if self.dictionary else []
        }


if __name__ == "__main__":
    b = DatabaseDictionary.load(DatabaseBook('demo'))

    print(b.to_json())
