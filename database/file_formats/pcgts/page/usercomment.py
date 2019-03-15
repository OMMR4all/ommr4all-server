from database.file_formats.pcgts.page.coords import Rect
from typing import List


class UserComment:
    def __init__(self, uc_id="", text="", aabb: Rect = None):
        self.id = uc_id
        self.text = text
        self.aabb = aabb

    @staticmethod
    def from_json(json: dict):
        if not json:
            return UserComment()

        return UserComment(
            json.get("id", ""),
            json.get("text", ""),
            Rect.from_json(json["aabb"]) if json.get('aabb', None) else None,
        )

    def to_json(self):
        return {
            "id": self.id,
            "text": self.text,
            "aabb": self.aabb.to_json() if self.aabb else None,
        }


class UserComments:
    def __init__(self, page, comments: List[UserComment] = None):
        self.page = page
        self.comments = comments if comments else []

    @staticmethod
    def from_json(json: dict, page):
        comments = UserComments(page)
        if json:
            comments.comments = [UserComment.from_json(c) for c in json.get('comments', [])]
        return comments

    def to_json(self):
        return {
            'comments': [c.to_json() for c in self.comments]
        }
