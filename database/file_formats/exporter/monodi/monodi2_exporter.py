import database.file_formats.pcgts as ns_pcgts
from typing import List, NamedTuple, Union, Optional
import json
import uuid
from enum import Enum
import numpy as np


class NoteType(Enum):
    Ascending = "Ascending"
    Descending = "Descending"
    Flat = "Flat"
    Liquescent = "Liquescent"
    Natural = "Natural"
    Normal = "Normal"
    Oriscus = "Oriscus"
    Quilisma = "Quilisma"
    Sharp = "Sharp"
    Strophicus = "Strophicus"

    @staticmethod
    def from_note(note: ns_pcgts.MusicSymbol):
        PNT = ns_pcgts.NoteType
        if note.note_type == PNT.NORMAL:
            return NoteType.Normal
        elif note.note_type == PNT.LIQUESCENT_FOLLOWING_U or note.note_type == PNT.LIQUESCENT_FOLLOWING_D:
            return NoteType.Liquescent
        elif note.note_type == PNT.ORISCUS:
            return NoteType.Oriscus
        elif note.note_type == PNT.APOSTROPHA:
            return NoteType.Strophicus

        return NoteType.Normal

    @staticmethod
    def from_accid(accid: ns_pcgts.MusicSymbol):
        AT = ns_pcgts.AccidType
        if accid.accid_type == AT.FLAT:
            return NoteType.Flat
        elif accid.accid_type == AT.SHARP:
            return NoteType.Sharp
        else:
            return NoteType.Natural


class BaseNote(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

    @staticmethod
    def from_name(name: ns_pcgts.NoteName):
        return BaseNote(str(name))

    @staticmethod
    def from_symbol(note: ns_pcgts.MusicSymbol):
        return BaseNote.from_name(note.note_name)


class Note(NamedTuple):
    noteType: NoteType
    base: BaseNote
    liquecent: bool
    octave: int
    focus: bool = False

    def to_json(self):
        return {
            'noteType': self.noteType.value,
            'base': self.base.value,
            'liquescent': self.liquecent,
            'octave': self.octave,
            'focus': self.focus,
            'uuid': str(uuid.uuid4()),
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


class LinePartKind(Enum):
    CLEF = "Clef"
    FOLIO_CHANGE = "FolioChange"
    LINE_CHANGE = "LineChange"
    SYLLABLE = "Syllable"


class Clef(NamedTuple):
    base: BaseNote
    octave: int
    shape: str

    def to_json(self):
        return {
            'uuid': str(uuid.uuid4()),
            'kind': LinePartKind.CLEF.value,
            'focus': False,
            'base': self.base.value,
            'octave': self.octave,
            'shape': self.shape

        }


class FolioChange:
    text: str

    def to_json(self):
        return {
            'uuid': str(uuid.uuid4()),
            'kind': LinePartKind.FOLIO_CHANGE.value,
            'focus': False,
            'text': self.text,

        }


class LineChange:
    def to_json(self):
        return {
            'uuid': str(uuid.uuid4()),
            'kind': LinePartKind.LINE_CHANGE.value,
            'focus': False,
        }


class SyllableType(Enum):
    EDITORIAL_ELLIPSIS = 'EditorialEllipsis'
    NORMAL = 'Normal'
    SOURCE_ELLIPSIS = 'SourceEllipsis'
    WITHOUT_NOTES = 'WithoutNotes'


class Syllable(NamedTuple):
    text: str
    notes: SpacedNotes
    syllableType: SyllableType = SyllableType.NORMAL

    def to_json(self):
        return {
            'uuid': str(uuid.uuid4()),
            'kind': LinePartKind.SYLLABLE.value,
            'text': self.text,
            'notes': self.notes.to_json(),
            'syllableType': self.syllableType.value,
        }


LinePart = Union[Clef, FolioChange, LineChange, Syllable]


class ContainerKind(Enum):
    FORM_CONTAINER = "FormteilContainer"
    MISC_CONTAINER = "MiscContainer"
    PARATEXT_CONTAINER = "ParatextContainer"
    ROOT_CONTAINER = "RootContainer"
    LINE_CONTAINER = "ZeileContainer"


class ParatextContainer(NamedTuple):
    text: str

    def to_json(self):
        return {
            'kind': ContainerKind.PARATEXT_CONTAINER.value,
            'uuid': str(uuid.uuid4()),
            'text': self.text,
            'retro': False,
            'paratextType': 'Formteil',
        }


class LineContainer(NamedTuple):
    children: List[LinePart]

    def to_json(self):
        return {
            'kind': ContainerKind.LINE_CONTAINER.value,
            'uuid': str(uuid.uuid4()),
            'children': [c.to_json() for c in self.children],
        }


MiscChildren = Union[LineContainer, ParatextContainer]


class MiscContainer(NamedTuple):
    children: List[MiscChildren]

    def to_json(self):
        return {
            'kind': ContainerKind.MISC_CONTAINER.value,
            'uuid': str(uuid.uuid4()),
            'children': [c.to_json() for c in self.children],
        }


FormChildren = Union[LineContainer, ParatextContainer, 'FormContainer']


class FormContainer(NamedTuple):
    children: List[FormChildren]

    def to_json(self):
        return {
            'kind': ContainerKind.FORM_CONTAINER.value,
            'uuid': str(uuid.uuid4()),
            'children': [c.to_json() for c in self.children],
            'data': None,
        }


RootChildren = Union[FormContainer, MiscContainer]

class RootContainer(NamedTuple):
    children: List[RootChildren]

    def to_json(self):
        return {
            'kind': ContainerKind.ROOT_CONTAINER.value,
            'uuid': str(uuid.uuid4()),
            'children': [c.to_json() for c in self.children],
            'comments': [],
            'documentType': 'Level1',
        }


Container = Union[FormContainer, MiscContainer, ParatextContainer, RootContainer, LineContainer]

"""

RootContainer:
    [
        MiscContainer:
            [
                LineContainer:
                    [
                        Syllable:
                            SpacedNotes:
                                [
                                    NonSpacedNotes:
                                        [
                                            GroupedNotes:
                                                [
                                                    Notes
                                                ]
                                        ]
                                ]
                        
                        ...
                        
                    ]
                    
                ParatextContainer
                    
                ...
            ]
    ]

"""


class PcgtsToMonodiConverter:
    def __init__(self, pcgts: List[ns_pcgts.PcGts], document=False):
        self.current_line_container: Optional[LineContainer] = None
        self.miscContainer = MiscContainer([])
        self.line_containers = self.miscContainer.children
        self.root = RootContainer([self.miscContainer])
        self.run(pcgts, documents=document)

    def get_or_create_current_line_container(self):
        if self.current_line_container is None:
            self.current_line_container = LineContainer([])
            self.line_containers.append(self.current_line_container)

        return self.current_line_container

    def get_or_create_syllables(self):
        clc = self.get_or_create_current_line_container()
        if len(clc.children) == 0 or not isinstance(clc.children[-1], Syllable):
            s = Syllable('', SpacedNotes([]))
            clc.children.append(s)
            return s
        else:
            return clc.children[-1]

    def run(self, pcgts: List[ns_pcgts.PcGts], documents=False):
        document_start = False
        regions_to_export = [ns_pcgts.BlockType.HEADING, ns_pcgts.BlockType.FOLIO_NUMBER, ns_pcgts.BlockType.PARAGRAPH, ns_pcgts.BlockType.MUSIC]
        for p in pcgts:
            elements: List[ns_pcgts.Block] = p.page.blocks_of_type(regions_to_export)

            elements.sort(key=lambda r: np.mean([line.coords.aabb().center.y for line in r.lines]))
            for element in elements:
                self.current_line_container = None

                if element.block_type == ns_pcgts.BlockType.MUSIC:

                    mr = element
                    connections = [c for c in p.page.annotations.connections if c.music_region == mr]
                    if documents:
                        tr = set(c.text_region for c in connections)
                        te = [te.lines[0].sentence.document_start for te in tr]
                        if True in te and not document_start:
                            document_start = True
                        elif True in te and document_start:
                            break
                        elif document_start:
                            pass
                        else:
                            continue

                    if len(connections) == 0:
                        continue

                    symbols = []
                    for s in mr.lines:
                        symbols += s.symbols

                    current_symbol_index = 0
                    if len(symbols) == 0:
                        continue

                    def add_line_symbols(line_symbols: List[ns_pcgts.MusicSymbol]):
                        clc = self.get_or_create_current_line_container()
                        current_syllable: Optional[Syllable] = None
                        for line_symbol in line_symbols:
                            if line_symbol.symbol_type != ns_pcgts.SymbolType.NOTE:
                                current_syllable = None

                            if line_symbol.symbol_type == ns_pcgts.SymbolType.CLEF:
                                clc.children.append(
                                    Clef(
                                        base=BaseNote.from_name(line_symbol.note_name),
                                        octave=line_symbol.octave,
                                        shape=line_symbol.clef_type.value.upper(),
                                    )
                                )
                            elif line_symbol.symbol_type == ns_pcgts.SymbolType.ACCID:
                                syllable = self.get_or_create_syllables()
                                syllable.notes.spaced.append(
                                    NonSpacesNotes([GroupedNotes([
                                        Note(
                                            base=BaseNote.from_name(line_symbol.note_name),
                                            octave=line_symbol.octave,
                                            noteType=NoteType.from_accid(line_symbol),
                                            liquecent=False,
                                        )
                                    ])])
                                )
                            elif line_symbol.symbol_type == ns_pcgts.SymbolType.NOTE:
                                if not current_syllable:
                                    current_syllable = self.get_or_create_syllables()

                                syllable = current_syllable
                                if len(syllable.notes.spaced) == 0 or line_symbol.graphical_connection == ns_pcgts.GraphicalConnectionType.NEUME_START:
                                    nsn = NonSpacesNotes([])
                                    syllable.notes.spaced.append(nsn)
                                else:
                                    nsn = syllable.notes.spaced[-1]

                                if line_symbol.graphical_connection == ns_pcgts.GraphicalConnectionType.LOOPED:
                                    nsn.non_spaced[-1].grouped.append(
                                        Note(
                                            base=BaseNote.from_symbol(line_symbol),
                                            octave=line_symbol.octave,
                                            noteType=NoteType.from_note(line_symbol),
                                            liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D, ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                        )
                                    )
                                else:
                                    gn = GroupedNotes([
                                        Note(
                                            base=BaseNote.from_symbol(line_symbol),
                                            octave=line_symbol.octave,
                                            noteType=NoteType.from_note(line_symbol),
                                            liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D, ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                        )
                                    ])
                                    nsn.non_spaced.append(gn)

                            else:
                                raise TypeError(type(line_symbol))

                    all_syllable_connections = sum([c.syllable_connections for c in connections], [])
                    all_syllable_connections.sort(key=lambda sc: sc.note.coord.x)
                    for sc in all_syllable_connections:
                        note = sc.note
                        try:
                            neume_pos = symbols.index(note, current_symbol_index)
                        except ValueError as e:
                            print(e)
                            continue
                        line_symbols = symbols[current_symbol_index:neume_pos]
                        # add symbols until position of connection
                        add_line_symbols(line_symbols)
                        current_symbol_index = neume_pos

                        # add the syllable
                        self.get_or_create_current_line_container().children.append(
                            Syllable(sc.syllable.text, SpacedNotes([]))
                        )

                    add_line_symbols(symbols[current_symbol_index:])
                else:
                    tr = element
                    text = " ".join([tl.text() for tl in tr.lines])
                    if len(text) == 0:
                        continue
                    self.line_containers.append(
                        ParatextContainer(
                            text=text
                        )
                    )
            else:
                continue
            break




if __name__=="__main__":
    from database import DatabaseBook
    b = DatabaseBook('demo2')
    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts')) for x in b.pages()]
    root = PcgtsToMonodiConverter(pcgts, document=True).root
    print(json.dumps(root.to_json(), indent=2))
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(root.to_json(), f, ensure_ascii=False, indent=4)
