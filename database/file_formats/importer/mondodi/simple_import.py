from typing import List

from database.file_formats.pcgts import MusicSymbol, SymbolType, NoteType, GraphicalConnectionType, NoteName
from database.file_formats.pcgts.page import Syllable, SyllableConnection


class MonodiDocument:
    def __init__(self, sentence):
        self.sentence = sentence

    def get_word_list(self, text_normalizer):
        words = []
        word = ""
        for x in self.sentence:
            if len(x) > 0:
                if x[-1] == "-":
                    word = word + x[:-1]
                elif x == "\n" or x == "\n\n":
                    pass
                else:
                    word = word + x
                    words.append(word)
                    word = ''
        return words

    def get_text(self, text_normalizer=None):
        text = ""
        for x in self.sentence:
            if len(x) > 0:
                text += x
                if x[-1] == "-":
                    pass
                else:
                    text += " "
        return text_normalizer.apply(text) if text_normalizer else text


def simple_monodi_data_importer1(json):
    sentence = []
    for x in json["children"]:
        for y in x["children"]:
            if y["kind"] == "ZeileContainer":
                for z in y["children"]:
                    if z["kind"] == "Syllable":
                        sentence.append(z["text"])
                    elif z["kind"] == "LineChange":
                        pass
                        # sentence.append("\n")
                    elif z["kind"] == "FolioChange":
                        pass
                        # sentence.append("\n")
                    else:
                        pass
                        # print(z["kind"])
    return MonodiDocument(sentence)


def getRowContainer(dict, list):
    if "children" in dict:
        for x in dict["children"]:
            if "kind" in x and x["kind"] != "ZeileContainer":
                getRowContainer(x, list=list)
            else:
                list.append(x)
    else:
        pass
        # print(dict)


def simple_monodi_data_importer(json):
    sentence = []
    row_container = []
    getRowContainer(json, row_container)
    for x in row_container:
        for z in x["children"]:
            if z["kind"] == "Syllable":
                sentence.append(z["text"])
            elif z["kind"] == "LineChange":
                pass
                sentence.append("\n")
            elif z["kind"] == "FolioChange":
                pass
                sentence.append("\n\n")
            else:
                pass
    return MonodiDocument(sentence)


def get_music_symbols(note_dict, ignore_liquescent = True):
    symbols: List[MusicSymbol] = []
    g_c = [GraphicalConnectionType.NEUME_START, GraphicalConnectionType.GAPED, GraphicalConnectionType.LOOPED]
    last = 0
    for spaced_note in note_dict["spaced"]:
        for non_spaced_note in spaced_note["nonSpaced"]:
            for grouped in non_spaced_note["grouped"]:

                if ignore_liquescent:
                    if not grouped["liquescent"]:
                        symbols.append(MusicSymbol(symbol_type=SymbolType("note"),
                                                   note_type=NoteType(0),
                                                   octave=grouped["octave"],
                                                   note_name=NoteName.from_string(grouped["base"]),
                                                   graphical_connection=g_c[last]))
                else:
                    symbols.append(MusicSymbol(symbol_type=SymbolType("note"),
                                               note_type=NoteType(0),
                                               octave=grouped["octave"],
                                               note_name=NoteName.from_string(grouped["base"]),
                                               graphical_connection=g_c[last]))
                last = 2
            last = 1
        last = 0
    return symbols


from dataclasses import dataclass


@dataclass
class Neume:
    symbols: List[MusicSymbol]
    syllable: Syllable


@dataclass
class Row:
    neumes: List[Neume]


def simple_monodi_data_importer2(json, ignore_liquescent=True) -> List[Row]:
    rows: List[Row] = []
    row_container = []
    getRowContainer(json, row_container)
    last_syllable = 0
    neumes: List[Neume] = []
    for x in row_container:
        for z in x["children"]:
            if z["kind"] == "Syllable":
                symbols = get_music_symbols(z["notes"], ignore_liquescent=ignore_liquescent)
                syllable = Syllable(text=z["text"],
                                    connection=SyllableConnection.NEW if last_syllable == 0 else SyllableConnection.HIDDEN)
                neumes.append(Neume(symbols=symbols, syllable=syllable))
                if len(z["text"]) > 0:
                    if z["text"][-1] == "-":
                        last_syllable = 1
                    else:
                        last_syllable = 0
                    #else:
                    #    last_syllable = 0 #?
            elif z["kind"] == "LineChange":
                rows.append(Row(neumes))
                neumes = []
            elif z["kind"] == "FolioChange":
                rows.append(Row(neumes))
                neumes = []
            else:
                pass
    rows.append(Row(neumes))

    return rows
