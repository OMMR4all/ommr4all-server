from dataclasses import dataclass

import database.file_formats.pcgts as ns_pcgts
from typing import List, NamedTuple, Union, Optional
import json
import uuid
from enum import Enum
import numpy as np

from database.file_formats.book.document import Document
from database.file_formats.pcgts import Page, Line, SymbolType
from database.file_formats.pcgts.page import SyllableConnector


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

@dataclass
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

@dataclass
class Syllable():
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
            'data': [
                {
                    "name": "Signatur",
                    "data": ""
                }
            ],
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


class FormatUrl:
    def __init__(self, url="https://iiif-ls6.informatik.uni-wuerzburg.de/iiif/3/"):
        self.base_url = "https://iiif-ls6.informatik.uni-wuerzburg.de/iiif/3/"

    def get_url(self, source, page, suffix=".jpg"):
        return self.base_url + source + "%2F" + page + suffix
    # url = "https://iiif-ls6.informatik.uni-wuerzburg.de/iiif/3/mul_2%2Ffolio_0202v.jpg/full/max/0/default.jpg"


class PcgtsToMonodiConverter:
    def __init__(self, pcgts: List[ns_pcgts.PcGts], document: Document = None, replace_filename="folio_"):
        self.current_line_container: Optional[LineContainer] = None
        self.miscContainer = FormContainer([])
        self.line_containers = self.miscContainer.children
        self.root = RootContainer([self.miscContainer])
        self.run2(pcgts, document=document, replace_fn=replace_filename)

    def get_Monodi_json(self, document: Document, editor):
        doc, notes = self.get_meta_and_notes(document, editor)
        return {"document": doc, "notes": notes}

    def get_meta_and_notes(self, document: Document, editor, replace="folio_",
                           url="https://iiif-ls6.informatik.uni-wuerzburg.de/iiif/3/", sourceIIF="mul_2",
                           doc_source="Mul 2", suffix=".jpg"):
        formatstring = FormatUrl(url)
        doc = {"id": document.monody_id,
               "quelle_id": doc_source,
               "dokumenten_id": document.monody_id,
               "gattung1": document.document_meta_infos.genre if document.document_meta_infos else "",
               "gattung2": "",
               "festtag": document.document_meta_infos.festum if document.document_meta_infos else "",
               "feier": document.document_meta_infos.dies if document.document_meta_infos else "",
               "textinitium": document.document_meta_infos.initium.replace("-",
                                                                           "") if document.document_meta_infos and document.document_meta_infos.initium and len(
                   document.document_meta_infos.initium) > 0 else document.textinitium.replace("-", ""),
               "bibliographischerverweis": "",
               "druckausgabe": "",
               "zeilenstart": str(document.start.row),
               "foliostart": document.start.page_name.replace(replace, ""),
               "kommentar": "",
               "editionsstatus": "",
               "additionalData": {
                   "Melodiennummer_Katalog": "",
                   "Editor": str(editor),
                   "Bezugsgesang": "",
                   "Melodie_Standard": "",
                   "Endseite": document.end.page_name.replace(replace, ""),
                   "Startposition": "",
                   "Zusatz_zu_Textinitium": "",
                   "Referenz_auf_Spiel": "",
                   "Endzeile": str(document.end.row),
                   "Nachtragsschicht": "",
                   "\u00dcberlieferungszustand": "",
                   "Melodie_Quelle": [formatstring.get_url(source=sourceIIF, page=i, suffix=suffix) for i in
                                      document.pages_names] if url else [],
                   "iiifs": [formatstring.get_url(source=sourceIIF, page=i, suffix=suffix) for i in
                             document.pages_names] if url else [],
                   "manuscript": document.document_meta_infos.manuscript if document.document_meta_infos else "",
               },
               "publish": None
               }
        return doc, self.root.to_json()

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

    def get_last_syllables(self):
        clc = self.get_or_create_current_line_container()
        if len(clc.children) == 0 or not isinstance(clc.children[-1], Syllable):
            return None
        else:
            return clc.children[-1]

    def create_folio_change(self, page):
        lc = self.get_current_line_container()
        if lc:
            f = FolioChange(text=page)
            lc.children.append(f)
        pass
    def get_current_line_container(self):
        if self.current_line_container is None:
            return None

        return self.current_line_container
    """
    def run2(self, pcgts: List[ns_pcgts.PcGts], document: Document = None):
        for i in r_or:
            symbols = [con.note for c in p.page.annotations.connections for con in c.syllable_connections if
                       con.syllable in i.sentence.syllables]
            current_symbol_index = 0

            if document is not None:
                line_id_start = document.start.line_id
                line_id_end = document.end.line_id

                # connection = connections[0]

                # line_ids = [line.id for line in connection.text_region.lines]
                if page.p_id == document.end.page_id:
                    if line_id_end == i.id:
                        break
                if page.p_id == document.start.page_id or document_started:

                    if line_id_start == i.id or document_started:
                        add_block(symbols)
                        document_started = True

            else:
                add_block(symbols)
        break
        """

    def run2(self, pcgts: List[ns_pcgts.PcGts], document: Document = None, replace_fn=""):
        if not document:
            self.run(pcgts, document)
        else:
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
                                    liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D,
                                                                        ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                )
                            )
                        else:
                            gn = GroupedNotes([
                                Note(
                                    base=BaseNote.from_symbol(line_symbol),
                                    octave=line_symbol.octave,
                                    noteType=NoteType.from_note(line_symbol),
                                    liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D,
                                                                        ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                )
                            ])
                            nsn.non_spaced.append(gn)

                    else:
                        raise TypeError(type(line_symbol))

            document_started = False
            regions_to_export = [ns_pcgts.BlockType.HEADING, ns_pcgts.BlockType.FOLIO_NUMBER,
                                 ns_pcgts.BlockType.PARAGRAPH,
                                 ns_pcgts.BlockType.MUSIC]
            stop = False
            last_syllable = None
            for p in pcgts:
                fn = p.page.location.page.replace(replace_fn, "") if len(replace_fn) > 0 else p.page.location.page
                self.create_folio_change(fn)
                elements: List[ns_pcgts.Block] = p.page.blocks_of_type(regions_to_export)
                page = p.page

                elements.sort(key=lambda r: np.mean([line.coords.aabb().center.y for line in r.lines]))

                r_or: List[Line] = page.reading_order.reading_order

                for i in r_or:
                    self.current_line_container = None

                    line_id_start = document.start.line_id
                    line_id_end = document.end.line_id
                    if page.p_id == document.end.page_id:
                        if line_id_end == i.id:
                            stop = True
                            break
                    if page.p_id == document.start.page_id or document_started:

                        if line_id_start == i.id or document_started:

                            document_started = True
                            connections_b = page.annotations.connections
                            connections_d = [c.music_region for c in connections_b for con in
                                             c.syllable_connections if
                                             con.syllable in i.sentence.syllables]

                            connections: List[SyllableConnector] = [con for c in connections_b for con in
                                                                    c.syllable_connections if
                                                                    con.syllable in i.sentence.syllables]
                            connections = sorted(connections, key=lambda x: x.note.coord.x)
                            music_block = page.closest_music_block_to_text_line(i)

                            all_connections = sorted(
                                sum([c.syllable_connections for c in p.page.annotations.connections if
                                     c.music_region == music_block], []),
                                key=lambda x: x.note.coord.x)
                            music_region = page.closest_music_line_to_text_line(i)
                            all_symbols = music_region.symbols

                            if len(connections) > 0:

                                note = connections[0].note
                                current_symbol_index = all_symbols.index(note)
                                first = True
                                for sc in connections:
                                    note = sc.note
                                    try:
                                        neume_pos = all_symbols.index(note)
                                    except ValueError as e:
                                        print(e)
                                        continue
                                    if first:
                                        types = [x.symbol_type for x in all_symbols[0:current_symbol_index]]
                                        ind = [ind for ind, i in enumerate(types) if i != SymbolType.NOTE]
                                        for i in ind[::-1]:
                                            if current_symbol_index - i == 1:
                                                current_symbol_index = i
                                            else:
                                                break

                                    line_symbols = all_symbols[current_symbol_index:neume_pos]
                                    add_line_symbols(line_symbols)
                                    current_symbol_index = neume_pos

                                    # add the syllable
                                    #ls = self.get_last_syllables()
                                    syllable = Syllable(sc.syllable.text ,
                                            SpacedNotes([]))
                                    if sc.syllable.connection != sc.syllable.connection.NEW:
                                        last_syllable.text = last_syllable.text + "-"
                                    self.get_or_create_current_line_container().children.append(syllable

                                    )
                                    last_syllable = syllable
                                    first = False
                                # new_start = [ind for ind, s in enumerate(all_symbols[current_symbol_index:]) if s.symbol_type == s.symbol_type.NOTE and s.graphical_connection == s.graphical_connection.NEUME_START][0] + current_symbol_index
                                index = all_connections.index(connections[-1])
                                if index == len(all_connections) - 1:
                                    new_start = len(all_symbols)
                                else:
                                    new_start = all_symbols.index(all_connections[index + 1].note)
                                add_line_symbols(all_symbols[current_symbol_index:new_start + 1])  # todo
                            else:
                                if len(all_connections) > 0:
                                    new_start = all_symbols.index(all_connections[0].note)
                                    add_line_symbols(all_symbols[0:new_start])  # todo

                                    pass
                                else:
                                    add_line_symbols(all_symbols)  # todo

                            """
                            nonlocal current_symbol_index
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
                            """

                if stop:
                    break

    def run(self, pcgts: List[ns_pcgts.PcGts], document: Document = None):
        def add_block(symbols):
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
                                    liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D,
                                                                        ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                )
                            )
                        else:
                            gn = GroupedNotes([
                                Note(
                                    base=BaseNote.from_symbol(line_symbol),
                                    octave=line_symbol.octave,
                                    noteType=NoteType.from_note(line_symbol),
                                    liquecent=line_symbol.note_type in [ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_D,
                                                                        ns_pcgts.NoteType.LIQUESCENT_FOLLOWING_U],
                                )
                            ])
                            nsn.non_spaced.append(gn)

                    else:
                        raise TypeError(type(line_symbol))

            nonlocal current_symbol_index
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
                    Syllable(
                        sc.syllable.text if sc.syllable.connection == sc.syllable.connection.NEW else "-" + sc.syllable.text,
                        SpacedNotes([]))
                )

            add_line_symbols(symbols[current_symbol_index:])

        document_started = False
        regions_to_export = [ns_pcgts.BlockType.HEADING, ns_pcgts.BlockType.FOLIO_NUMBER, ns_pcgts.BlockType.PARAGRAPH,
                             ns_pcgts.BlockType.MUSIC]
        for p in pcgts:
            elements: List[ns_pcgts.Block] = p.page.blocks_of_type(regions_to_export)
            page = p.page

            elements.sort(key=lambda r: np.mean([line.coords.aabb().center.y for line in r.lines]))

            def elements_sort_music_blocks_by_reading_order(elements: List[ns_pcgts.Block], page_: Page):
                sorted = []
                rest = []
                r_or = page_.reading_order.reading_order
                m_ids = []
                for i in r_or:
                    ml = page_.closest_music_line_to_text_line(i)
                    if ml.id not in m_ids:
                        stop = False
                        for block in elements:
                            for bl in block.lines:
                                if bl.id == ml.id:
                                    sorted.append(block)
                                    stop = True
                                    break
                            if stop:
                                break

                        m_ids.append(ml.id)

                return sorted + rest

            elements = elements_sort_music_blocks_by_reading_order(elements, page)
            for element in elements:
                self.current_line_container = None

                if element.block_type == ns_pcgts.BlockType.MUSIC:

                    mr = element
                    connections = [c for c in p.page.annotations.connections if c.music_region == mr]
                    if len(connections) == 0:
                        continue

                    symbols = []
                    for s in mr.lines:
                        symbols += s.symbols

                    current_symbol_index = 0
                    if len(symbols) == 0:
                        continue
                    if document is not None:
                        line_id_start = document.start.line_id
                        line_id_end = document.end.line_id
                        connection = connections[0]

                        line_ids = [line.id for connection in connections for line in connection.text_region.lines]
                        if page.p_id == document.end.page_id:
                            if line_id_end in line_ids:
                                break
                        if page.p_id == document.start.page_id or document_started:

                            if line_id_start in line_ids or document_started:
                                add_block(symbols)
                                document_started = True

                    else:
                        add_block(symbols)
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


if __name__ == "__main__":
    from database import DatabaseBook

    b = DatabaseBook('demo2')
    pcgts = [ns_pcgts.PcGts.from_file(x.file('pcgts')) for x in b.pages()]
    root = PcgtsToMonodiConverter(pcgts, document=True).root
    print(json.dumps(root.to_json(), indent=2))
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(root.to_json(), f, ensure_ascii=False, indent=4)
