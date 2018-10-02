from datetime import datetime


class Meta:
    def __init__(self, creator="", created=datetime.now(), last_change=datetime.now()):
        self.creator = creator
        self.created = created
        self.last_change = last_change

    @staticmethod
    def from_json(json: dict):
        return Meta(
            json.get('creatpr', ""),
            json.get('created', ""),
            json.get('lastChange', ""),
        )

    def to_json(self):
        return {
            "creator": self.creator,
            "created": str(self.created),
            "lastChange": str(self.last_change),
        }

