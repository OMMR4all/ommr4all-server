from enum import IntEnum
from rest_framework.response import Response


class ErrorCodes(IntEnum):
    # Book related
    BOOK_EXISTS = 41001
    BOOK_INVALID_NAME = 41002
    BOOK_INSUFFICIENT_RIGHTS = 41003

    # Page related
    PAGE_EXISTS = 44001
    PAGE_INVALID_NAME = 44002
    PAGE_NOT_LOCKED = 44003


class APIError:
    def __init__(self,
                 status,
                 developer_message: str,
                 user_message: str,
                 error_code: ErrorCodes
                 ):
        self.status = status
        self.developer_message = developer_message
        self.user_message = user_message
        self.error_code = error_code

    def to_json(self):
        return {
            'status': self.status,
            'developerMessage': self.developer_message,
            'userMessage': self.user_message,
            'error_code': self.error_code.value,
        }

    def response(self):
        return Response(self.to_json(), status=self.status)


class PageNotLockedAPIError(APIError):
    def __init__(self, status):
        super().__init__(status, 'Page not locked by user. Access denied.',
                         'Access denied. You did not request access to the page',
                         ErrorCodes.PAGE_NOT_LOCKED
                         )

