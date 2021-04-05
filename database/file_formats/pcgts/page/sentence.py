from typing import List
from .syllable import Syllable, SyllableConnection
import re


class Sentence:
    def __init__(self,
                 syllables: List[Syllable]):
        self.syllables = syllables

    def text(self, with_drop_capital=True):
        t = ''
        for syllable in self.syllables:
            text = syllable.text
            if not with_drop_capital and syllable.drop_capital_length > 0:
                text = text[syllable.drop_capital_length:]

            if syllable.connection == SyllableConnection.NEW:
                if len(t) == 0:
                    t += text
                else:
                    t += ' ' + text
            elif syllable.connection == SyllableConnection.VISIBLE:
                t += '~' + text
            else:
                t += '-' + text
        return t

    def syllable_by_id(self, syllable_id):
        for s in self.syllables:
            if s.id == syllable_id:
                return s

        return None


    @staticmethod
    def from_json(json: dict):
        return Sentence(
            [s for s in [Syllable.from_json(s) for s in json.get('syllables', [])] if len(s.text) > 0]
        )

    # syllable_re = re.compile(r"(([\w.!?,;]+[~\-])|([\w.!?,;]+$))")
    syllable_re = re.compile(r"([~\-]?[\w.!?,;]+)")

    @staticmethod
    def from_string(text: str):
        words = text.split()
        syllables = []
        for word in words:
            for s in Sentence.syllable_re.findall(word):
                if s.startswith("-"):
                    syllables.append(Syllable(text=s[1:], connection=SyllableConnection.HIDDEN))
                elif s.startswith("~"):
                    syllables.append(Syllable(text=s[1:], connection=SyllableConnection.VISIBLE))
                else:
                    syllables.append(Syllable(text=s, connection=SyllableConnection.NEW))

        return Sentence(syllables)

    def to_json(self) -> dict:
        return {
            'syllables': [s.to_json() for s in self.syllables],
        }

