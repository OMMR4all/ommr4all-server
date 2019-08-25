from typing import DefaultDict, NamedTuple
import json
from enum import Enum
from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin


class Locks(Enum):
    STAFF_LINES = 'StaffLines'
    LAYOUT = 'Layout'
    SYMBOLS = 'Symbols'
    TEXT = 'Text'

    VERIFIED = 'Verified'

    @staticmethod
    def user_locks():
        return [Locks.STAFF_LINES, Locks.LAYOUT, Locks.SYMBOLS, Locks.TEXT]


class LockState(NamedTuple):
    label: str
    lock: bool


LockedStates = DefaultDict[Locks, bool]


@dataclass
class PageProgress(DataClassJSONMixin):
    locked: LockedStates = field(default_factory=lambda: LockedStates())

    @staticmethod
    def from_json_file(file: str):
        with open(file) as f:
            return PageProgress.from_json(f.read())

    def to_json_file(self, filename: str):
        s = json.dumps(self.to_dict(), indent=2)
        with open(filename, 'w') as f:
            f.write(s)

    def verified_allowed(self):
        return all([self.locked[l] for l in Locks.user_locks()])
