from enum import IntEnum
from rest_framework.response import Response
from mashumaro import DataClassDictMixin
from dataclasses import dataclass


class ErrorCodes(IntEnum):
    # Global
    UNKNOWN_ERROR = 1000

    CONNECTION_TO_SERVER_TIMED_OUT = 10001

    # Global permissions
    GLOBAL_INSUFFICIENT_RIGHTS = 21003

    # book style related
    BOOK_STYLE_INVALID_REQUEST = 23001

    # Book related
    BOOK_EXISTS = 41001
    BOOK_INVALID_NAME = 41002
    BOOK_INSUFFICIENT_RIGHTS = 41003

    BOOK_PAGES_RENAME_REQUIRE_UNIQUE_SOURCES = 41030
    BOOK_PAGES_RENAME_REQUIRE_UNIQUE_TARGETS = 41031
    BOOK_PAGES_RENAME_TARGET_EXISTS = 41032

    # Page related
    PAGE_EXISTS = 44001
    PAGE_INVALID_NAME = 44002
    PAGE_NOT_LOCKED = 44003

    # Page progress
    PAGE_PROGRESS_VERIFICATION_REQUIRES_ALL_PROGRESS_LOCKS = 44101

    # Operation related
    OPERATION_INVALID_GET = 50001
    OPERATION_UNKNOWN_SERVER_ERROR = 50002

    # Task related
    OPERATION_TASK_NOT_FOUND = 51001
    OPERATION_TASK_NO_MODEL = 51002

    # Task training related
    OPERATION_TASK_TRAIN_EMPTY_DATASET = 52001

    # Model related
    MODEL_NOT_FOUND = 55000


@dataclass
class APIError(DataClassDictMixin):
    status: int
    developerMessage: str
    userMessage: str
    errorCode: ErrorCodes

    def response(self):
        return Response(self.to_dict(), status=self.status)


class PageNotLockedAPIError(APIError):
    def __init__(self, status):
        super().__init__(status, 'Page not locked by user. Access denied.',
                         'Access denied. You did not request access to the page',
                         ErrorCodes.PAGE_NOT_LOCKED
                         )

