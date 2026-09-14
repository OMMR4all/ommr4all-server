import unittest
from django.test import TestCase as DjangoTestCase

from database.file_formats.pcgts.page.musicsymbol import (
    create_clef, ClefType, MusicSymbolPositionInStaff, NoteName, MusicSymbol, SymbolType,
    GraphicalConnectionType, NoteType,
)
from restapi.models.error import ErrorCodes


class TestGClef(unittest.TestCase):
    """The G clef is a built-in clef subtype without its own trainable label."""

    def test_name_octave(self):
        # the G clef marks the G above the C that a C clef marks at the same position
        clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.LINE_2), (NoteName.G, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.SPACE_3), (NoteName.A, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.SPACE_2), (NoteName.F, 5))
        self.assertTupleEqual(clef.note_name_octave(MusicSymbolPositionInStaff.LINE_1), (NoteName.E, 5))

        c_clef = create_clef(clef_type=ClefType.C, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertTupleEqual(c_clef.note_name_octave(MusicSymbolPositionInStaff.LINE_2), (NoteName.C, 5))

    def test_json_round_trip(self):
        clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        d = clef.to_json()
        self.assertEqual(d['type'], 'clef')
        self.assertEqual(d['clefType'], 'g')
        restored = MusicSymbol.from_json(d)
        self.assertEqual(restored.symbol_type, SymbolType.CLEF)
        self.assertEqual(restored.clef_type, ClefType.G)
        self.assertEqual(restored.position_in_staff, MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(restored.id, clef.id)


class TestUnsupportedSymbolClassTolerance(unittest.TestCase):
    """Workflow components must skip symbol classes they do not support
    instead of raising."""

    def test_pixel_classifier_label_mapping(self):
        from omr.imageoperations.music_line_operations import SymbolLabel
        from omr.imageoperations.symbol_label_set import SymbolClassLabelSets

        sets = SymbolClassLabelSets.builtin()
        g_clef = create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(sets.main_index_of(g_clef), 0)

        c_clef = create_clef(clef_type=ClefType.C, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(sets.main_index_of(c_clef), SymbolLabel.CLEF_C.value)
        f_clef = create_clef(clef_type=ClefType.F, position_in_staff=MusicSymbolPositionInStaff.LINE_2)
        self.assertEqual(sets.main_index_of(f_clef), SymbolLabel.CLEF_F.value)

    def test_calamari_codec_encodes_new_subtype(self):
        from omr.dataset.datastructs import CalamariCodec, CalamariSequence

        codec = CalamariCodec()
        symbols = [
            create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2),
            MusicSymbol(SymbolType.NOTE, position_in_staff=MusicSymbolPositionInStaff.SPACE_3),
        ]
        seq = CalamariSequence(codec, symbols)
        # both symbols are representable: the dynamic codec learns the new clef subtype
        self.assertEqual(len(seq.calamari_str), 2)

    def test_evaluator_sequence_skips_unknown_types(self):
        from omr.steps.symboldetection.evaluator import Codec
        # symbols_to_label_sequence must not raise for any defined subtype
        codec = Codec()
        symbols = [
            create_clef(clef_type=ClefType.G, position_in_staff=MusicSymbolPositionInStaff.LINE_2),
            MusicSymbol(SymbolType.NOTE, position_in_staff=MusicSymbolPositionInStaff.SPACE_3),
        ]
        seq = codec.symbols_to_label_sequence(symbols, False)
        self.assertEqual(len(seq), 2)


class TestRegisteredSymbolClass(DjangoTestCase):
    """A class registered at runtime round-trips through PCGTS and extends the
    label inventory of the notation style it is scoped to."""

    def test_clef_based_class_extends_the_main_head(self):
        from database.models.symbolclasses import SymbolClass
        from omr.imageoperations.music_line_operations import SymbolLabel
        from omr.imageoperations.symbol_label_set import symbol_label_sets_for_style

        SymbolClass.objects.create(id='custodes', name='Custodes', style=None,
                                   base_symbol_type='clef', base_sub_type='c', clef_offset=-3,
                                   glyph_preset='hook', order=10)
        sets = symbol_label_sets_for_style('french14')
        self.assertEqual(len(sets.main), 10)
        self.assertEqual(sets.main[9].class_id, 'custodes')
        self.assertEqual(len(sets.note_type), 11)

        s = create_clef(ClefType.C, symbol_class='custodes')
        self.assertEqual(MusicSymbol.from_json(s.to_json()).symbol_class, 'custodes')
        self.assertEqual(sets.main_index_of(s), 9)
        self.assertNotIn(sets.main[9].color, [l.get_color() for l in SymbolLabel])
        # the registered offset overrides the base clef's
        self.assertEqual(s.clef_offset(), -3)

    def test_note_based_class_extends_the_note_type_head(self):
        from database.models.symbolclasses import SymbolClass
        from omr.imageoperations.symbol_heads import AdditionalSymbolLabel
        from omr.imageoperations.symbol_label_set import symbol_label_sets_for_style

        SymbolClass.objects.create(id='quilisma', name='Quilisma', style=None,
                                   base_symbol_type='note', base_sub_type='0',
                                   glyph_preset='cross', order=10)
        sets = symbol_label_sets_for_style('french14')
        self.assertEqual(len(sets.main), 9)
        self.assertEqual(len(sets.note_type), 12)
        self.assertEqual(sets.note_type[11].class_id, 'quilisma')
        self.assertNotIn(sets.note_type[11].color, [l.get_color() for l in AdditionalSymbolLabel])

        n = MusicSymbol(SymbolType.NOTE, note_type=NoteType.NORMAL,
                        graphical_connection=GraphicalConnectionType.LOOPED,
                        symbol_class='quilisma')
        # the main head still predicts the graphical connection of a custom note class
        self.assertEqual(sets.main_index_of(n), 2)
        self.assertEqual(sets.note_type_index_of(n), 11)
        self.assertEqual(MusicSymbol.from_json(n.to_json()).symbol_class, 'quilisma')

    def test_unknown_class_id_is_preserved_and_degrades_to_its_base(self):
        from omr.imageoperations.symbol_label_set import symbol_label_sets_for_style

        sets = symbol_label_sets_for_style('french14')
        s = create_clef(ClefType.C, symbol_class='never_registered')
        self.assertEqual(MusicSymbol.from_json(s.to_json()).symbol_class, 'never_registered')
        # no registered label -> the base class label
        self.assertEqual(sets.main_index_of(s), 4)
        self.assertEqual(s.clef_offset(), ClefType.C.offset())


class TestSymbolClassesRestApi(DjangoTestCase):
    """The /api/symbol-classes surface: read is public, writing needs the new permissions."""

    CUSTODES = {
        'name': 'Custodes', 'style': None, 'base_symbol_type': 'clef', 'base_sub_type': 'c',
        'clef_offset': -3, 'glyph_preset': 'hook', 'svg_path': '', 'svg_path_stroke': None,
        'color': '', 'digit_shortcut': 8, 'hidden_by_default': False, 'order': 10,
    }

    def setUp(self):
        from django.contrib.auth.models import User
        self.user = User.objects.create_user('symbol_class_user', password='pw')
        self.admin = User.objects.create_superuser('symbol_class_admin', password='pw')

    def _client(self, user=None):
        from rest_framework.test import APIClient
        client = APIClient()
        if user is not None:
            client.force_authenticate(user=user)
        return client

    def test_create_list_and_delete(self):
        client = self._client(self.admin)
        response = client.put('/api/symbol-classes', self.CUSTODES, format='json')
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()['id'], 'Custodes')

        listed = self._client().get('/api/symbol-classes')
        self.assertEqual(listed.status_code, 200, listed.content)
        self.assertEqual([c['id'] for c in listed.json()], ['Custodes'])

        deleted = client.delete('/api/symbol-classes/Custodes')
        self.assertEqual(deleted.status_code, 200, deleted.content)
        self.assertEqual(self._client().get('/api/symbol-classes').json(), [])

    def test_duplicate_name_is_rejected(self):
        client = self._client(self.admin)
        client.put('/api/symbol-classes', self.CUSTODES, format='json')
        response = client.put('/api/symbol-classes', self.CUSTODES, format='json')
        self.assertEqual(response.status_code, 400, response.content)
        self.assertEqual(response.json()['errorCode'], ErrorCodes.SYMBOL_CLASS_EXISTS.value)

    def test_a_builtin_digit_shortcut_is_rejected(self):
        body = dict(self.CUSTODES, digit_shortcut=2)
        response = self._client(self.admin).put('/api/symbol-classes', body, format='json')
        self.assertEqual(response.status_code, 400, response.content)
        self.assertEqual(response.json()['errorCode'],
                         ErrorCodes.SYMBOL_CLASS_DIGIT_SHORTCUT_TAKEN.value)

    def test_an_unparseable_base_class_is_rejected(self):
        for body in [dict(self.CUSTODES, base_sub_type='not_a_clef'),
                     dict(self.CUSTODES, base_symbol_type='neume'),
                     dict(self.CUSTODES, base_symbol_type='note', base_sub_type='9')]:
            response = self._client(self.admin).put('/api/symbol-classes', body, format='json')
            self.assertEqual(response.status_code, 400, response.content)
            self.assertEqual(response.json()['errorCode'],
                             ErrorCodes.SYMBOL_CLASS_INVALID_REQUEST.value)

    def test_an_empty_name_is_rejected(self):
        response = self._client(self.admin).put('/api/symbol-classes', dict(self.CUSTODES, name=''),
                                                format='json')
        self.assertEqual(response.status_code, 400, response.content)
        self.assertEqual(response.json()['errorCode'], ErrorCodes.SYMBOL_CLASS_INVALID_NAME.value)

    def test_editing_keeps_its_own_digit_shortcut(self):
        client = self._client(self.admin)
        client.put('/api/symbol-classes', self.CUSTODES, format='json')
        response = client.post('/api/symbol-classes/Custodes',
                               dict(self.CUSTODES, glyph_preset='diamond'), format='json')
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(self._client().get('/api/symbol-classes').json()[0]['glyph_preset'],
                         'diamond')

    def test_writing_requires_the_permission(self):
        client = self._client(self.user)
        self.assertEqual(client.put('/api/symbol-classes', self.CUSTODES, format='json').status_code,
                         401)
        self.assertEqual(client.delete('/api/symbol-classes/Custodes').status_code, 401)
        self.assertEqual(client.post('/api/symbol-classes/Custodes', self.CUSTODES,
                                     format='json').status_code, 401)


if __name__ == '__main__':
    unittest.main()
