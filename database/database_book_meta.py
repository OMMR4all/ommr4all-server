from dataclasses import dataclass, field

import dateutil
from mashumaro import field_options
from mashumaro.types import SerializationStrategy

from database.database_book import DatabaseBook
import os
from database.database_internal import DEFAULT_MODELS
from datetime import datetime
# from mashumaro import DataClassJSONMixin
from mashumaro.mixins.json import DataClassJSONMixin

from typing import Any, Optional, Dict, List
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmTypes
from restapi.models.auth import RestAPIUser
from dateutil import parser


def get_default_book_style():
    from database.models.bookstyles import DEFAULT_BOOK_STYLE
    return DEFAULT_BOOK_STYLE


class FormattedDateTime(SerializationStrategy):
    def serialize(self, value: datetime) -> str:
        return value.isoformat()

    def deserialize(self, value: str) -> datetime:
        return parser.parse(value)


@dataclass
class DatabaseBookMeta(DataClassJSONMixin):
    id: str = ''
    name: str = ''
    created: datetime = field(default_factory=lambda: datetime.now(),
                              metadata=field_options(serialization_strategy=FormattedDateTime()))
    # timestamp and user of the last content modification (page/pcgts/progress writes); None for legacy books
    updated: Optional[datetime] = field(default=None,
                                        metadata=field_options(serialization_strategy=FormattedDateTime()))
    updatedBy: Optional[str] = None
    creator: Optional[RestAPIUser] = None
    last_opened: str = ''
    notationStyle: str = field(default_factory=lambda: get_default_book_style())
    numberOfStaffLines: int = 4
    algorithmPredictorParams: Dict[AlgorithmTypes, AlgorithmPredictorParams] = field(default_factory=lambda: {})
    dateOfOrigin: str = ''
    placeOfOrigin: str = ''
    # configured one-click workflow of the client; the server only stores it
    oneClickWorkflow: List[Dict[str, Any]] = field(default_factory=list)

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
    def from_book_json(book: DatabaseBook, json: str):
        from database.models.bookstyles import BookStyle, DEFAULT_BOOK_STYLE
        meta = DatabaseBookMeta.from_json(json)
        meta.id = book.book
        if len(meta.name) == 0:
            meta.name = book.book

        try:
            BookStyle.objects.get(id=meta.notationStyle)
        except BookStyle.DoesNotExist:
            meta.notationStyle = DEFAULT_BOOK_STYLE

        return meta

    def to_file(self, book: DatabaseBook):
        import tempfile
        self.id = book.book
        s = self.to_json()
        path = book.local_path('book_meta.json')
        # atomic replace: concurrent page saves of the same book both bump the meta,
        # a reader must never observe a partially written file
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.book_meta.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(s)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        from database.book_index import safe_index_book_meta
        safe_index_book_meta(book)


if __name__ == '__main__':
    b = DatabaseBookMeta.load(DatabaseBook('Graduel'))
    print(b.to_json())
