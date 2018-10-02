from enum import Enum
import re
from typing import List
from uuid import uuid4
from omr.datatypes.page.definitions import EquivIndex


class SyllableConnection(Enum):
    VISIBLE = 'visible'     # Dashes, e.g. Ex-am-ple
    HIDDEN = 'hidden'       # No dashes visible, e.g. Example
    NEW = 'new'             # New word, e.g. The bat


class Syllable:
    syllable_re = re.compile("(([\w\.!?,;]+[~\-])|([\w\.!?,;]+$))")

    @staticmethod
    def syllables_from_textequiv(text_equiv):
        words = text_equiv.content.split()
        syllables = []
        sub_id = 0
        for word in words:
            for s, _, _ in Syllable.syllable_re.findall(word):
                s_id = "{}:{}".format(text_equiv.id, sub_id)
                if s.endswith("-"):
                    syllables.append(Syllable(s_id, s[:-1], SyllableConnection.HIDDEN))
                elif s.endswith("~"):
                    syllables.append(Syllable(s_id, s[:-1], SyllableConnection.VISIBLE))
                else:
                    syllables.append(Syllable(s_id, s, SyllableConnection.NEW))

                sub_id += 1

        return syllables

    def __init__(self, s_id=str(uuid4()), text="", connection=SyllableConnection.NEW):
        self.id = str(s_id)
        self.text = text
        self.connection = connection

    @staticmethod
    def from_json(json: dict, s_id: str):
        return Syllable(
            s_id,
            json.get('text', ""),
            SyllableConnection[json.get('connection', SyllableConnection.NEW)],
        )

    def to_json(self):
        return {
            'text': self.text,
            'connection': self.connection.value
        }


class TextEquiv:
    def __init__(self, t_id=str(uuid4()), content="", index=EquivIndex.GROUND_TRUTH):
        self.id = str(t_id)
        self.content = content
        self.index = index
        self._syllables: List[Syllable] = None

    def syllables(self) -> List[Syllable]:
        if self._syllables is None:
            self._syllables = Syllable.syllables_from_textequiv(self)

        return self._syllables

    def syllable_by_id(self, syllable_id):
        for s in self.syllables():
            if s.id == syllable_id:
                return s

        return None

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
            "syllables": [s.to_json() for s in self.syllables()],
        }


if __name__ == '__main__':
    te1 = TextEquiv(content="Die Ver-bin-dung muss hier auf~ge-löst wer~den!")
    print(te1.to_json())
    print(TextEquiv.from_json(te1.to_json()).to_json() == te1.to_json())
    te2 = TextEquiv(content="Die-ser Satz! Und die~se, Sät~ze?")
    print(te2.to_json())
    print(TextEquiv.from_json(te2.to_json()).to_json() == te2.to_json())
    print(Syllable.syllables_from_textequiv(TextEquiv(content="Das ist ein Test.")))
    print(Syllable.syllables_from_textequiv(TextEquiv(content="Die Ver-bind-dung muss hier auf~ge-löst wer~den!")))

