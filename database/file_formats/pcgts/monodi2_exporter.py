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
    def from_note(note: ns_pcgts.NoteComponent):
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
    def from_accid(accid: ns_pcgts.Accidental):
        AT = ns_pcgts.AccidentalType
        if accid.accidental == AT.FLAT:
            return NoteType.Flat
        elif accid.accidental == AT.SHARP:
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
    def from_clef(clef: ns_pcgts.Clef):
        name, _ = clef.note_name_octave(clef.position_in_staff)
        return BaseNote.from_name(name)

    @staticmethod
    def from_note(note: ns_pcgts.NoteComponent):
        return BaseNote.from_name(note.note_name)

    @staticmethod
    def from_accid(note: ns_pcgts.NoteComponent):
        return BaseNote(str(note.note_name))


class Note(NamedTuple):
    noteType: NoteType
    base: BaseNote
    liquecent: bool
    octave: int
    focus: bool = False
    uuid: str = str(uuid.uuid4())

    def to_json(self):
        return {
            'noteType': self.noteType.value,
            'base': self.base.value,
            'liquescent': self.liquecent,
            'octave': self.octave,
            'focus': self.focus,
            'uuid': self.uuid,
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
            'documentType': 'Gesang',
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
    def __init__(self, pcgts: List[ns_pcgts.PcGts]):
        self.current_line_container: Optional[LineContainer] = None
        self.miscContainer = MiscContainer([])
        self.line_containers = self.miscContainer.children
        self.root = RootContainer([self.miscContainer])

        self.run(pcgts)

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

    def run(self, pcgts: List[ns_pcgts.PcGts]):
        text_regions_to_export = [ns_pcgts.TextRegionType.HEADING, ns_pcgts.TextRegionType.FOLIO_NUMBER, ns_pcgts.TextRegionType.PARAGRAPH]
        for p in pcgts:
            elements: List[Union[ns_pcgts.MusicRegion, ns_pcgts.TextRegion]] \
                = p.page.music_regions + [tr for tr in p.page.text_regions if tr.region_type in text_regions_to_export]

            elements.sort(key=lambda r: np.mean([line.coords.aabb().center.y for line in r.children]))
            for element in elements:
                self.current_line_container = None

                if isinstance(element, ns_pcgts.TextRegion):
                    tr: ns_pcgts.TextRegion = element
                    text = " ".join([tl.text() for tl in tr.text_lines])
                    if len(text) == 0:
                        continue
                    self.line_containers.append(
                        ParatextContainer(
                            text=text
                        )
                    )
                elif isinstance(element, ns_pcgts.MusicRegion):
                    mr: ns_pcgts.MusicRegion = element
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

                    def add_line_symbols(line_symbols):
                        clc = self.get_or_create_current_line_container()
                        for line_symbol in line_symbols:
                            if isinstance(line_symbol, ns_pcgts.Clef):
                                clef: ns_pcgts.Clef = line_symbol
                                clc.children.append(
                                    Clef(
                                        base=BaseNote.from_name(line_symbol.note_name),
                                        octave=line_symbol.octave,
                                        shape="FC"[clef.clef_type.value],
                                    )
                                )
                            elif isinstance(line_symbol, ns_pcgts.Accidental):
                                accid: ns_pcgts.Accidental = line_symbol
                                syllable = self.get_or_create_syllables()
                                syllable.notes.spaced.append(
                                    NonSpacesNotes([GroupedNotes([
                                        Note(
                                            base=BaseNote.from_name(accid.note_name),
                                            octave=accid.octave,
                                            noteType=NoteType.from_accid(accid),
                                            liquecent=False,
                                        )
                                    ])])
                                )
                            elif isinstance(line_symbol, ns_pcgts.Neume):
                                nsn = NonSpacesNotes([])
                                syllable = self.get_or_create_syllables()
                                syllable.notes.spaced.append(nsn)
                                for i, nc in enumerate(line_symbol.notes):
                                    if i > 0 and nc.graphical_connection == ns_pcgts.GraphicalConnectionType.LOOPED:
                                        nsn.non_spaced[-1].grouped.append(
                                            Note(
                                                base=BaseNote.from_note(nc),
                                                octave=nc.octave,
                                                noteType=NoteType.from_note(nc),
                                                liquecent=nc.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D, ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                            )
                                        )
                                    else:
                                        gn = GroupedNotes([
                                            Note(
                                                base=BaseNote.from_note(nc),
                                                octave=nc.octave,
                                                noteType=NoteType.from_note(nc),
                                                liquecent=nc.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D, ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                            )
                                        ])
                                        nsn.non_spaced.append(gn)

                            else:
                                raise TypeError(type(line_symbol))

                    for sc in c.syllable_connections:

                        if len(sc.neume_connections) == 0:
                            continue

                        neume = sc.neume_connections[0].neume
                        try:
                            neume_pos = symbols.index(neume, current_symbol_index)
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


if __name__=="__main__":
    from database import DatabaseBook
    b = DatabaseBook('Graduel')
    pcgts = [ns_pcgts.PcGts.from_file(p.file('pcgts')) for p in b.pages()[4:5]]
    root = PcgtsToMonodiConverter(pcgts).root
    print(json.dumps(root.to_json(), indent=2))
