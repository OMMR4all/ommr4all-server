from functools import lru_cache
from io import StringIO

import database.file_formats.pcgts as ns_pcgts
from typing import Iterable, List, NamedTuple, Union, Optional
from lxml import etree
import numpy as np
from database.file_formats.book.document import symbols_of_document_staff
from database.file_formats.exporter.mei.neume_dict import NeumeDict
from enum import Enum
import logging
import os
from ommr4all.settings import BASE_DIR
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _relax_ng(schema_path: str) -> etree.RelaxNG:
    """The MEI schema is >1 MB and parsing it dominates a per-document export."""
    return etree.RelaxNG(file=schema_path)


class PcgtsToMeiConverter:
    def __init__(self, pcgts: ns_pcgts.PcGts, document_line_ids: Optional[Iterable[str]] = None):
        # None: export the whole page (unchanged behaviour). A collection of text line ids:
        # export only the staves and syllables of that document (see document_export.py).
        self.document_line_ids: Optional[set] = set(document_line_ids) if document_line_ids is not None else None
        self.document_lines_of_staff = {}
        self.has_content = False
        self.neume_container = []
        self.previous_symbol_value = None
        self.neume_dict = NeumeDict()
        self.doc = None
        self.root = None
        self.music = None
        self.score = None  # Page
        self.scoreDef = None
        self.section = None
        self.staff = None  # current staff
        self.layer = None  # layer
        self.neume = None
        self.syllable = None
        self.body = None
        self.mdiv = None
        self.init()
        self.convert(pcgts)

        # an empty <mdiv> never validates; such a page is dropped by the caller anyway
        self.is_valid = self.has_content and self.validate(
            os.path.join(BASE_DIR, 'database', 'file_formats', 'exporter', 'mei', 'mei-all.rng'))

    def init(self):
        self.root = etree.Element("mei", meiversion="4.0.1", xmlns="http://www.music-encoding.org/ns/mei")
        self.doc = etree.ElementTree(self.root)
        head = etree.SubElement(self.root, 'meiHead')
        file_desc = etree.SubElement(head, 'fileDesc')
        title_stmt = etree.SubElement(file_desc, 'titleStmt')
        title = etree.SubElement(title_stmt, 'title').text = 'MEI Encoding Output'
        title_stmt = etree.SubElement(file_desc, 'pubStmt')

        self.music = etree.SubElement(self.root, 'music')
        self.body = etree.SubElement(self.music, 'body')
        self.mdiv = etree.SubElement(self.body, 'mdiv')

    def convert(self, pcgts):
        regions_to_export = [ns_pcgts.BlockType.MUSIC, ns_pcgts.BlockType.HEADING,
                             ns_pcgts.BlockType.FOLIO_NUMBER, ns_pcgts.BlockType.PARAGRAPH]

        self.score = etree.SubElement(self.mdiv, 'score')

        elements: List[ns_pcgts.Block] = pcgts.page.blocks_of_type(regions_to_export)
        elements.sort(key=lambda r: np.mean([line.coords.aabb().center.y for line in r.lines]))
        elements, music_lines_of_block = self._filter_to_document(pcgts.page, elements)
        self.has_content = len(elements) > 0
        staff_counter = 0
        if len(elements) > 0:
            self.scoreDef = etree.SubElement(self.score, 'scoreDef')
            staffGrp = etree.SubElement(self.scoreDef, 'staffGrp')
            self.section = etree.SubElement(self.score, 'section')

        for element in elements:
            if element.block_type == ns_pcgts.BlockType.MUSIC:
                staff_counter += 1
                music_lines = music_lines_of_block[id(element)]
                staffDef = etree.SubElement(staffGrp, 'staffDef', lines=str(len(music_lines[0].staff_lines)),
                                            n=str(staff_counter), notationtype="neume")
                self.add_staff(staff_counter)
                mr = element
                symbols = []
                for s in music_lines:
                    symbols += self._symbols_of_music_line(pcgts.page, s)
                current_symbol_index = 0
                connections = [c for c in pcgts.page.annotations.connections if c.music_region == mr]
                if len(connections) != 0:
                    all_syllable_connections = sum([c.syllable_connections for c in connections], [])
                    all_syllable_connections = self._filter_connections_to_document(all_syllable_connections)
                    all_syllable_connections.sort(key=lambda sc: sc.note.coord.x)
                    for sc in all_syllable_connections:
                        note = sc.note
                        try:
                            neume_pos = symbols.index(note, current_symbol_index)
                        except ValueError as e:
                            continue
                        line_symbols = symbols[current_symbol_index:neume_pos]
                        current_symbol_index = neume_pos
                        for symbol in line_symbols:
                            self.add_node_symbols(symbol)
                        self.add_syllable(sc.syllable.text)
                    for symbol in symbols[current_symbol_index:]:
                        self.add_node_symbols(symbol)
                else:
                    for symbol in symbols:
                        self.add_node_symbols(symbol)
            else:
                tr = element
                self.add_accompanying_text(tr)

    def _filter_to_document(self, page, elements: List['ns_pcgts.Block']):
        """Reduce the page's blocks to what belongs to the document.

        Returns the blocks to export and, per music block, the music lines to export.
        Without a document this is a no-op that keeps every block and every line.
        """
        music_lines_of_block = {id(b): b.lines for b in elements}
        if self.document_line_ids is None:
            return elements, music_lines_of_block

        # a lyric line of the document pins the staff above it
        self.document_lines_of_staff = {}
        for text_line in page.reading_order.reading_order:
            if text_line.id not in self.document_line_ids:
                continue
            music_line = page.closest_music_line_to_text_line(text_line)
            if music_line is not None:
                self.document_lines_of_staff.setdefault(music_line.id, []).append(text_line)

        kept = []
        for block in elements:
            # accompanying text (headings, folio numbers, paragraphs) is not part of the
            # reading order, so it cannot be attributed to a chant; the Monodi+ document
            # export drops it too and the two exports have to agree
            if block.block_type != ns_pcgts.BlockType.MUSIC:
                continue
            lines = [line for line in block.lines if line.id in self.document_lines_of_staff]
            if len(lines) == 0:
                continue
            music_lines_of_block[id(block)] = lines
            kept.append(block)
        return kept, music_lines_of_block

    def _symbols_of_music_line(self, page, music_line):
        if self.document_line_ids is None:
            return music_line.symbols
        return symbols_of_document_staff(page, music_line, self.document_line_ids)

    def _filter_connections_to_document(self, syllable_connections):
        if self.document_line_ids is None:
            return syllable_connections
        syllables = [syllable
                     for text_lines in self.document_lines_of_staff.values()
                     for text_line in text_lines
                     for syllable in text_line.sentence.syllables]
        return [sc for sc in syllable_connections if sc.syllable in syllables]

    def add_accompanying_text(self, text_block):
        div = etree.SubElement(self.section, 'div')
        lg = etree.SubElement(div, 'lg')
        for tl in text_block.lines:
            text = tl.text()
            l = etree.SubElement(lg, 'l').text = text

    def add_staff(self, staff_counter):
        self.staff = etree.SubElement(self.section, 'staff', n=str(staff_counter))
        self.layer = etree.SubElement(self.staff, 'layer', n='1')

    def add_syllable(self, text):
        self.syllable = etree.SubElement(self.layer, 'syllable')
        syl = etree.SubElement(self.syllable, 'syl').text = text

    def add_node_symbols(self, symbol: ns_pcgts.MusicSymbol):
        x, y = symbol.coord.xy()
        if symbol.symbol_type not in (ns_pcgts.SymbolType.NOTE, ns_pcgts.SymbolType.CLEF, ns_pcgts.SymbolType.ACCID):
            # tolerate symbol types this exporter does not support
            return
        if symbol.symbol_type == ns_pcgts.SymbolType.CLEF:
            clef = etree.SubElement(self.layer, 'clef', oct=str(symbol.octave),
                                             shape=str(symbol.note_name), line=str(symbol.position_in_staff // 2))
            return
        if symbol.symbol_type == ns_pcgts.SymbolType.ACCID:
            accid = etree.SubElement(self.layer, 'accid', accid=str(symbol.accid_type).lower(),
                                     x=str(x), y=str(y))
            return

        symbol_note_type = symbol.note_type
        if self.syllable is None:

            self.syllable = etree.SubElement(self.layer, 'syllable')

        def cvsymbol_type(symbol_type):
            cv_list = {
                'NORMAL': None,
                'ORISCUS': 'oriscus',
                'APOSTROPHA': 'apostropha',
                'LIQUESCENT_FOLLOWING_U': None,
                'LIQUESCENT_FOLLOWING_D': None,
            }
            return cv_list[symbol_type]

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
        d = {'oct': str(symbol.octave - 1),
             'pname': str(symbol.note_name).lower(),
             'con': 'g' if symbol.graphical_connection == symbol.graphical_connection.GAPED else 'l'
             if symbol.graphical_connection == symbol.graphical_connection.LOOPED else None,
             'x': str(x),
             'y': str(y),
             # 'type': cvsymbol_type(symbol.note_type.name),
             }

        d = dict((k, v) for k, v in d.items() if v)

        if self.neume is None or symbol.graphical_connection is symbol.graphical_connection.NEUME_START:
            '''
            neume_type = self.neume_dict.get_neume_type(self.neume_container)
            if self.neume is not None and neume_type is not None:
                self.neume.attrib['type'] = self.neume_dict.get_neume_type(self.neume_container)
            '''
            self.neume_container = []
            self.neume = etree.SubElement(self.syllable, 'neume')
            self.previous_symbol_value = symbol.note_name.value

        nc = etree.SubElement(self.neume, 'nc', **d)

        enum = {0: '', -1: 'up', 1: 'down'}
        c_v = enum.get(min(enum.keys(), key=lambda i: abs(i - (self.previous_symbol_value - symbol.note_name.value))))
        self.neume_container.append(c_v + symbol.graphical_connection.name.lower())
        self.previous_symbol_value = symbol.note_name.value

    def validate(self, schema_path: str) -> bool:

        z = etree.tostring(self.doc, pretty_print=True)
        z = z.decode("utf-8")
        test = etree.parse(StringIO(z))
        relaxNG = _relax_ng(schema_path)

        result = relaxNG.validate(test)
        if result:
            logger.debug('Validated as {} with MEI4.0.1 Shemata '.format(result))
        else:
            logger.warning(relaxNG.error_log)

        return result

    def write(self, fp, pretty_print=True):
        self.doc.write(fp, pretty_print=pretty_print)

    def to_string(self,  pretty_print=True):
        return etree.tostring(self.doc, pretty_print=pretty_print).decode('utf-8')


if __name__ == "__main__":
    from database import DatabaseBook
    b = DatabaseBook('demo')
    pcgts = [p.pcgts() for p in b.pages()]
    for p in pcgts:
        print(p.page.location.local_path())
        c = PcgtsToMeiConverter(p)
        # print(c.to_string())
