import json
from collections import defaultdict
from typing import List, NamedTuple


class ActionHistoryEntry(NamedTuple):
    action: str
    time: int


ActionHistory = List[ActionHistoryEntry]


class Statistics:
    @staticmethod
    def from_json_file(filename):
        with open(filename) as f:
            return Statistics.from_json(json.load(f))

    @staticmethod
    def from_json(d):
        s = Statistics()
        for key, value in d.get('actions', {}).items():
            s.actions[key] = int(value)

        for key, value in d.get('toolTiming', {}).items():
            s.tool_timing[key] = int(value)

        for value in d.get('actionHistory', []):
            s.action_history.append(ActionHistoryEntry(value['action'], value['time']))

        return s

    def to_json_file(self, filename):
        s = json.dumps(self.to_json(), indent=2)
        with open(filename, 'w') as f:
            f.write(s)

    def to_json(self):
        return {
            'actions': self.actions,
            'toolTiming': self.tool_timing,
            'actionHistory': [a._asdict() for a in self.action_history],
        }

    def __init__(self):
        self.actions = defaultdict(int)
        self.tool_timing = defaultdict(int)
        self.action_history: ActionHistory = []

    def add(self, o: 'Statistics'):
        for key, val in o.actions.items():
            self.actions[key] += val

        for key, val in o.tool_timing.items():
            self.tool_timing[key] += val

        self.action_history += o.action_history

        return self


if __name__ == '__main__':
    s = Statistics.from_json({'toolTiming': {'TOOL1': 10, 'TOOL2': 20}})
    print(json.dumps(s.to_json(), indent=2))

