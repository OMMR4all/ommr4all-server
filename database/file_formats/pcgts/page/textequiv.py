from uuid import uuid4
from database.file_formats.pcgts.page import EquivIndex


class TextEquiv:
    def __init__(self, t_id=str(uuid4()), content="", index=EquivIndex.GROUND_TRUTH):
        self.id = str(t_id)
        self.content = content
        self.index = index

    @staticmethod
    def from_json(json):
        return TextEquiv(
            json.get("id", uuid4()),
            json.get("content", ""),
            EquivIndex(json.get("index", EquivIndex.GROUND_TRUTH)),
        )

    def to_json(self):
        return {
            "id": self.id,
            "content": self.content,
            "index": self.index.value,
        }


if __name__ == '__main__':
    te1 = TextEquiv(content="Die Ver-bin-dung muss hier auf~ge-löst wer~den!")
    print(te1.to_json())
    print(TextEquiv.from_json(te1.to_json()).to_json() == te1.to_json())
    te2 = TextEquiv(content="Die-ser Satz! Und die~se, Sät~ze?")
    print(te2.to_json())
    print(TextEquiv.from_json(te2.to_json()).to_json() == te2.to_json())

