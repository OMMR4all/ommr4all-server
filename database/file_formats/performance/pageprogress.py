from typing import DefaultDict, NamedTuple
import json
from enum import Enum
from dataclasses import dataclass, field
from mashumaro import DataClassJSONMixin
import logging

logger = logging.getLogger(__name__)


class Locks(Enum):
    STAFF_LINES = 'StaffLines'
    LAYOUT = 'Layout'
    SYMBOLS = 'Symbols'
    TEXT = 'Text'


class LockState(NamedTuple):
    label: Locks
    lock: bool


LockedStates = DefaultDict[Locks, bool]


@dataclass
class PageProgress(DataClassJSONMixin):
    locked: LockedStates = field(default_factory=lambda: LockedStates())
    verified: bool = False

    @staticmethod
    def from_json_file(file: str):
        with open(file) as f:
            try:
               pp = PageProgress.from_json(f.read())
            except Exception as e:
                logger.error("Exception when parsing file {}. Creating empty page progress".format(file))
                logging.exception(e)
                pp = PageProgress()

            pp.consistency_check()
            return pp

    def to_json_file(self, filename: str):
        self.consistency_check()
        s = json.dumps(self.to_dict(), indent=2)
        with open(filename, 'w') as f:
            f.write(s)

    def merge_local(self, p: 'PageProgress', locks=True, verified=True) -> 'PageProgress':
        if locks:
            for key, value in p.locked.items():
                self.locked[key] = value

        if verified:
            self.verified = p.verified

        self.consistency_check()
        return self

    def verified_allowed(self):
        return all([self.locked.get(l, False) for l in Locks])

    def consistency_check(self):
        if not self.verified_allowed():
            self.verified = False
