import unittest
from database.file_formats.pcgts.page.musicsymbol import create_clef, ClefType, MusicSymbolPositionInStaff, NoteName, MusicSymbol, SymbolType
from collections import namedtuple

class TestClefPIS(unittest.TestCase):

    def test_clef_pis(self):
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.F), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_3, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.G), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_3, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.A), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.B), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.C), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.D), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.E), ((9, 5), (23, 19)))

        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.A), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_3, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.B), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_3, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.C), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.D), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.E), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.F), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.G), ((7, 3), (21, 17)))

        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_3, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.A), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.B), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_4, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.C), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.D), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_5, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.E), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.SPACE_6, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.F), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_6, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(NoteName.G), ((9, 5), (23, 19)))

        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.A), ((7, 3), (21, 17)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.B), ((-1, -5), (13, 9)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.C), ((5, 1), (19, 15)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.D), ((-3, -7), (11, 7)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.E), ((3, -1), (17, 13)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.F), ((9, 5), (23, 19)))
        x = MusicSymbol(position_in_staff=MusicSymbolPositionInStaff.LINE_2, symbol_type=SymbolType.NOTE)
        self.assertEqual(x.pis_octave(note_name=NoteName.G), ((1, -3), (15, 11)))



if __name__ == '__main__':
    unittest.main()