from typing import DefaultDict, NamedTuple
import json


class LockState(NamedTuple):
    label: str
    lock: bool


class LockedStates(DefaultDict[str, bool]):
    def __init__(self):
        super().__init__(int)


class PageProgress:
    @staticmethod
    def from_json(d: dict):
        pp = PageProgress()
        for key, value in d.get('locked', {}).items():
            pp.locked[key] = bool(value)

        return pp

    def to_json(self):
        return {
            'locked': self.locked,
        }

    @staticmethod
    def from_json_file(file: str):
        with open(file) as f:
            return PageProgress.from_json(json.load(f))

    def to_json_file(self, filename: str):
        s = json.dumps(self.to_json(), indent=2)
        with open(filename, 'w') as f:
            f.write(s)

    def __init__(self):
        self.locked = LockedStates()
