from omr.datatypes import PcGts, TextRegionType, ClefType, Clef, Accidental, AccidentalType, NoteComponent, Neume, MusicSymbolPositionInStaff, GraphicalConnectionType
from typing import List, NamedTuple, Dict
import json


class Note(NamedTuple):
    base: str
    liquescent: bool
    noteType: str
    octave: int
    focus: bool = False

    def to_json(self):
        return {
            'base': self.base,
            'liquescent': self.liquescent,
            'noteType': self.noteType,
            'octave': self.octave,
            'focus': self.focus,
        }


class GroupedNotes(NamedTuple):
    grouped: List[Note]

    def to_json(self):
        return {'grouped': [s.to_json() for s in self.grouped]}


class NonSpacesNotes(NamedTuple):
    non_spaced: List[GroupedNotes]

    def to_json(self):
        return {'nonSpaced': [s.to_json() for s in self.non_spaced]}


class SpacedNotes(NamedTuple):
    spaced: List[NonSpacesNotes]

    def to_json(self):
        return {'spaced': [s.to_json() for s in self.spaced]}


class TextContainer(NamedTuple):
    text: str
    notes: SpacedNotes

    def to_json(self):
        return {'text': self.text, 'notes': self.notes.to_json()}


class LineContainer(NamedTuple):
    children: List[TextContainer]

    def to_json(self):
        return {'kind': 'ZeileContainer', 'children': [c.to_json() for c in self.children]}


class FormContainer(NamedTuple):
    children: List[LineContainer]

    def to_json(self):
        return {'kind': 'FormteilContainer', 'children': [c.to_json() for c in self.children]}


class RootContainer(NamedTuple):
    children: List[FormContainer]

    def to_json(self):
        return {'kind': 'RootContainer', 'children': [c.to_json() for c in self.children]}


def pcgts_to_monodi(pcgts: List[PcGts]) -> RootContainer:
    form = FormContainer([])
    root = RootContainer([form])

    for p in pcgts:
        for mr in p.page.music_regions:
            c = [c for c in p.page.annotations.connections if c.music_region == mr]
            if len(c) == 0:
                continue

            symbols = []
            for s in mr.staffs:
                symbols += s.symbols

            current_symbol_index = 0
            if len(symbols) == 0:
                continue

            c = c[0]
            text_containers = [TextContainer('', SpacedNotes([]))]

            def add_line_symbols(line_symbols):
                for line_symbol in line_symbols:
                    if isinstance(line_symbol, Clef):
                        pass
                    elif isinstance(line_symbol, Accidental):
                        pass
                    elif isinstance(line_symbol, Neume):
                        nsn = NonSpacesNotes([])
                        text_containers[-1].notes.spaced.append(nsn)
                        for nc in line_symbol.notes:
                            if nc.graphical_connection == GraphicalConnectionType.LOOPED:
                                nsn.non_spaced[-1].grouped.append(
                                    Note(
                                        str(nc.note_name), False, '-', nc.octave, False
                                    )
                                )
                            else:
                                gn = GroupedNotes([
                                    Note(
                                        str(nc.note_name), False, '-', nc.octave, False
                                    )
                                ])
                                nsn.non_spaced.append(gn)

                    else:
                        raise TypeError(type(line_symbol))

            for sc in c.syllable_connections:

                if len(sc.neume_connections) == 0:
                    continue

                neume = sc.neume_connections[0].neume
                neume_pos = symbols.index(neume, current_symbol_index)
                line_symbols = symbols[current_symbol_index:neume_pos]
                current_symbol_index = neume_pos

                add_line_symbols(line_symbols)

                syllable = TextContainer(sc.syllable.text, SpacedNotes([]))
                text_containers.append(syllable)

            add_line_symbols(symbols[current_symbol_index:])

            line = LineContainer([c for c in text_containers if len(c.notes.spaced) > 0])

            if len(line.children) > 0:
                form.children.append(line)

    return root


if __name__=="__main__":
    import main.book as book
    b = book.Book('test')
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:1]]
    print(json.dumps(pcgts_to_monodi(pcgts).to_json(), indent=2))
