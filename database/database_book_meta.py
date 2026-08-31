from dataclasses import dataclass, field

import dateutil
from mashumaro import field_options
from mashumaro.types import SerializationStrategy

from database.database_book import DatabaseBook
import os
from database.database_internal import DEFAULT_MODELS
from database.file_formats.pcgts.page.pitchparams import PitchDetectionParams
from datetime import datetime
# from mashumaro import DataClassJSONMixin
from mashumaro.mixins.json import DataClassJSONMixin

from typing import Any, Optional, Dict, List
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmTypes
from restapi.models.auth import RestAPIUser
from dateutil import parser
import logging

logger = logging.getLogger(__name__)


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
    # tolerances of the on-line/in-space decision of every symbol of this book, see PitchDetectionParams
    pitchDetectionParams: PitchDetectionParams = field(default_factory=lambda: PitchDetectionParams())
    algorithmPredictorParams: Dict[AlgorithmTypes, AlgorithmPredictorParams] = field(default_factory=lambda: {})
    dateOfOrigin: str = ''
    placeOfOrigin: str = ''
    # Monodi+ export: how this book is addressed in the Corpus Monodicum editor and on a IIIF
    # image server. Empty iiifImageApi/iiifSource means no image urls are exported at all,
    # which is the honest default — a guessed url points at somebody else's manuscript.
    monodiSourceId: str = ''
    iiifImageApi: str = ''
    iiifSource: str = ''
    iiifSuffix: str = '.jpg'
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
        self.id = book.book
        # atomic replace: concurrent page saves of the same book both bump the meta,
        # a reader must never observe a partially written file
        from database.file_write import write_text_atomic
        write_text_atomic(book.local_path('book_meta.json'), self.to_json())
        from database.book_index import safe_index_book_meta
        safe_index_book_meta(book)


# Pages are constructed in tight loops (training, book wide operations) and every one of them
# needs the pitch parameters of its book, while get_meta() re-reads book_meta.json on each call.
# Cache by mtime so an edit in the settings takes effect without a restart.
_pitch_params_cache = {}


def pitch_params_of_book(book: Optional[DatabaseBook]) -> PitchDetectionParams:
    if book is None:
        return PitchDetectionParams()

    path = book.local_path('book_meta.json')
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return PitchDetectionParams()

    cached = _pitch_params_cache.get(book.book)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        params = DatabaseBookMeta.load(book).pitchDetectionParams.clamped()
    except Exception:
        # the parameters must never be the reason a page fails to load
        logger.exception("Could not read the pitch detection parameters of book {}".format(book.book))
        params = PitchDetectionParams()

    _pitch_params_cache[book.book] = (mtime, params)
    return params


if __name__ == '__main__':
    b = DatabaseBookMeta.load(DatabaseBook('Graduel'))
    print(b.to_json())
