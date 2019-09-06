from dataclasses import dataclass, field
from database.database_book import DatabaseBook
import json
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
from mashumaro import DataClassJSONMixin
from typing import Optional, Dict, Tuple
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmTypes
from restapi.api.auth import RestAPIUser


@dataclass
class DatabaseBookMeta(DataClassJSONMixin):
    id: str = ''
    name: str = ''
    created: datetime = field(default_factory=lambda: datetime.now())
    creator: Optional[RestAPIUser] = None
    last_opened: str = ''
    notationStyle: str = 'french14'
    numberOfStaffLines: int = 4
    algorithmPredictorParams: Dict[AlgorithmTypes, AlgorithmPredictorParams] = field(default_factory=lambda: {})

    def algorithm_predictor_params(self, algorithm_type: AlgorithmTypes) -> AlgorithmPredictorParams:
        params = self.algorithmPredictorParams.get(algorithm_type, AlgorithmPredictorParams())

        # default values
        min_sl = params.minNumberOfStaffLines if params.minNumberOfStaffLines else self.numberOfStaffLines
        max_sl = params.maxNumberOfStaffLines if params.maxNumberOfStaffLines else self.numberOfStaffLines

        params.maxNumberOfStaffLines = max(min_sl, max_sl)
        params.minNumberOfStaffLines = min(min_sl, max_sl)

        return params

    def default_models_path(self):
        return os.path.join(DEFAULT_MODELS, self.notationStyle)

    @staticmethod
    def load(book: DatabaseBook):
        path = book.local_path('book_meta.json')
        try:
            with open(path) as f:
                d = DatabaseBookMeta.from_book_json(book, f.read())
        except FileNotFoundError:
            d = DatabaseBookMeta(id=book.book, name=book.book)

        return d

    @staticmethod
    def from_book_dict(book: DatabaseBook, json: dict):
        meta = DatabaseBookMeta.load(book)
        for key, value in json.items():
            setattr(meta, key, value)

        return meta

    @staticmethod
    def from_book_json(book: DatabaseBook, json: str):
        meta = DatabaseBookMeta.from_json(json)
        meta.id = book.book
        if len(meta.name) == 0:
            meta.name = book.book

        return meta

    def to_file(self, book: DatabaseBook):
        self.id = book.book
        s = self.to_json(indent=2)
        with open(book.local_path('book_meta.json'), 'w') as f:
            f.write(s)


if __name__ == '__main__':
    b = DatabaseBookMeta.load(DatabaseBook('Graduel'))
    print(b.to_json())
