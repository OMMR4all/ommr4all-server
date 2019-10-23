from enum import Enum
import re
from uuid import uuid4


class SyllableConnection(Enum):
    VISIBLE = 0     # Dashes, e.g. Ex-am-ple
    HIDDEN  = 1     # No dashes visible, e.g. Example
    NEW     = 2     # New word, e.g. The bat


class Syllable:
    def __init__(self,
                 s_id=None,
                 text="",
                 connection=SyllableConnection.NEW,
                 drop_capital_length=0):
        self.id = s_id if s_id else str(uuid4())
        self.text = text
        self.connection = connection
        self.drop_capital_length = drop_capital_length

    @staticmethod
    def from_json(json: dict):
        return Syllable(
            json.get('id', None),
            json.get('text', ""),
            SyllableConnection(json.get('connection', SyllableConnection.NEW)),
            json.get('dropCapitalLength', 0),
        )

    def to_json(self):
        return {
            'id': self.id,
            'text': self.text,
            'connection': self.connection.value,
            'dropCapitalLength': self.drop_capital_length
        }

