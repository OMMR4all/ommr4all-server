import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from database.file_formats.pcgts import BlockType, MusicSymbolPositionInStaff as Pis, SymbolType, ClefType, \
    NoteType, GraphicalConnectionType, Point, LineStartType
from database.file_formats.pcgts.page import Block, Line, MusicSymbol, Page, Sentence, StaffLine, StaffLines, \
    Syllable, SyllableConnection
from database.file_formats.pcgts.page.annotations import Annotations, Connection, SyllableConnector
from database.file_formats.pcgts.page.coords import Coords
from omr.end2end.codec.codec import codec_lines, merge_codec_lines, write_codec
from omr.end2end.codec.data_generator import process_single_block
from omr.end2end.codec.dataset import SmartTokenizer
from omr.imageoperations.symbol_label_set import SymbolLabelSpec
from omr.steps.end2end.model_io import load_pretrained
from omr.steps.end2end.networks import End2EndNetwork
from omr.steps.end2end.predictor import Chunk, End2EndPredictor, apply_text_lines, assign_chunks_to_lines, \
    lyric_line_results, parse_chunks, plan_line_starts
from omr.steps.end2end.sample_builder import lyric_lines_for_music_block

CUSTODES = SymbolLabelSpec(index=9, id='custodes', color=[0, 0, 0], symbol_type=SymbolType.CLEF, sub_type='f',
                           class_id='custodes')
QUILISMA = SymbolLabelSpec(index=6, id='quilisma', color=[0, 0, 0], symbol_type=SymbolType.NOTE, sub_type='1',
                           class_id='quilisma')
BAR_LINE = SymbolLabelSpec(index=10, id='bar_line', color=[0, 0, 0], symbol_type=SymbolType.OTHER, sub_type='',
                           class_id='bar_line')
SPECS = {s.class_id: s for s in (CUSTODES, QUILISMA, BAR_LINE)}


def note(pis, gc=GraphicalConnectionType.NEUME_START, symbol_class=None, note_type=NoteType.NORMAL):
    return MusicSymbol(SymbolType.NOTE, position_in_staff=pis, graphical_connection=gc, note_type=note_type,
                       symbol_class=symbol_class)


def block_with_syllables(symbols, anchors):
    """A music block whose symbols `anchors` (indices) carry one syllable each."""
    block = Block(BlockType.MUSIC, lines=[Line(symbols=symbols)])
    text_region = Block(BlockType.LYRICS, lines=[Line()])
    connection = Connection(block, text_region, [
        SyllableConnector(Syllable(text=f"s{i}", connection=SyllableConnection.NEW), symbols[i]) for i in anchors])
    return block, [connection]


class TestRegisteredClassTokens(unittest.TestCase):
    def setUp(self):
        self.symbols = [
            MusicSymbol(SymbolType.CLEF, clef_type=ClefType.C, position_in_staff=Pis.LINE_3),
            note(Pis.SPACE_2),
            note(Pis.LINE_2, GraphicalConnectionType.LOOPED, symbol_class='quilisma', note_type=NoteType.ORISCUS),
            MusicSymbol(SymbolType.OTHER, position_in_staff=Pis.LINE_2, symbol_class='bar_line'),
            note(Pis.SPACE_3),
            MusicSymbol(SymbolType.CLEF, clef_type=ClefType.F, position_in_staff=Pis.LINE_2, symbol_class='custodes'),
        ]
        self.block, self.connections = block_with_syllables(self.symbols, [1, 4])

    def test_class_symbols_round_trip_through_the_codec(self):
        t = process_single_block(None, self.block, self.connections, set(SPECS))
        self.assertEqual(t.symbols, self.symbols)
        chunks = parse_chunks(t.text, class_specs=SPECS)
        parsed = [s for c in chunks for s in c.symbols]
        self.assertEqual([(s.symbol_type, s.symbol_class, s.position_in_staff) for s in parsed],
                         [(s.symbol_type, s.symbol_class, s.position_in_staff) for s in self.symbols])
        # a class symbol comes back as its registered base class
        self.assertEqual(parsed[2].note_type, NoteType.ORISCUS)
        self.assertEqual(parsed[2].graphical_connection, GraphicalConnectionType.LOOPED)
        self.assertEqual(parsed[5].clef_type, ClefType.F)
        # every token of the transcription is in the codec
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'codec.txt')
            write_codec(path, codec_lines(['s', '0', '1', '4'], class_specs=SPECS.values()))
            self.assertNotIn(SmartTokenizer(path).UNK, SmartTokenizer(path).encode(t.text))

    def test_without_registration_classes_fall_back_to_their_base_and_other_is_dropped(self):
        t = process_single_block(None, self.block, self.connections, set())
        self.assertEqual(t.symbols, [s for s in self.symbols if s.symbol_type != SymbolType.OTHER])
        parsed = [s for c in parse_chunks(t.text) for s in c.symbols]
        self.assertTrue(all(s.symbol_class is None for s in parsed))
        self.assertEqual(parsed[4].clef_type, ClefType.F)

    def test_locations_reach_their_symbols_behind_markers(self):
        t = process_single_block(None, self.block, self.connections, set(SPECS))
        text = "[NEW_CHANT] " + t.text
        # the decoder reports locations by character offset of the symbol token
        offsets = [i for i in range(len(text)) if text.startswith('(', i)]
        locs = {o: (i / 10, 0.5) for i, o in enumerate(offsets)}
        chunks = parse_chunks(text, class_specs=SPECS, locs=locs)
        self.assertEqual([l for c in chunks for l in c.locs], [(i / 10, 0.5) for i in range(len(offsets))])
        self.assertEqual([c.text for c in chunks], ['', 's1', 's4'])


def rect(left, top, right, bottom) -> Coords:
    return Coords(np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float))


def syllable_at(text, x, start=LineStartType.NONE):
    """A syllable chunk whose (only) note was placed at page x."""
    return Chunk(text, SyllableConnection.NEW, [MusicSymbol(SymbolType.NOTE, coord=Point(x, 0.1))], [None], start)


class TestLyricsOfSplitLines(unittest.TestCase):
    """A chant starting mid-line leaves one music line above two lyric lines; every syllable
    must land on the lyric line below its note."""

    def setUp(self):
        self.left, self.right = Line(coords=rect(0.05, 0.2, 0.30, 0.25)), Line(coords=rect(0.32, 0.2, 0.60, 0.25))
        self.lines = [self.left, self.right]

    def test_syllables_follow_their_notes_and_never_move_back(self):
        chunks = [Chunk('', SyllableConnection.NEW, [MusicSymbol(SymbolType.CLEF, coord=Point(0.06, 0.1))], [None]),
                  syllable_at('a', 0.10), syllable_at('b', 0.29),
                  syllable_at('c', 0.315),  # in the gap, nearer to the right line
                  syllable_at('d', 0.20),   # mislocated to the left: stays on the right line
                  Chunk('e', SyllableConnection.HIDDEN, [], []),
                  syllable_at('f', 0.70)]   # right of everything: nearest is the right line
        self.assertEqual(assign_chunks_to_lines(chunks, self.lines), [0, 0, 1, 1, 1, 1])

    def test_one_lyric_line_takes_every_syllable(self):
        chunks = [syllable_at('a', 0.5), syllable_at('b', 0.1)]
        self.assertEqual(assign_chunks_to_lines(chunks, [self.left]), [0, 0])

    def test_a_starting_syllable_just_before_the_next_line_joins_it(self):
        # the first note of a chant often sits above the drop capital, left of its lyric line
        early = [syllable_at('a', 0.10), syllable_at('b', 0.28, LineStartType.CHANT)]
        self.assertEqual(assign_chunks_to_lines(early, self.lines, start_tolerance=0.05), [0, 1])
        unmarked = [syllable_at('a', 0.10), syllable_at('b', 0.28)]
        self.assertEqual(assign_chunks_to_lines(unmarked, self.lines, start_tolerance=0.05), [0, 0])
        far = [syllable_at('a', 0.10), syllable_at('b', 0.20, LineStartType.CHANT)]
        self.assertEqual(assign_chunks_to_lines(far, self.lines, start_tolerance=0.05), [0, 0])

    def test_candidates_are_the_lines_in_the_row_below_the_music_left_to_right(self):
        music = Block(BlockType.MUSIC, lines=[Line(coords=rect(0.05, 0.1, 0.60, 0.19))])
        right_block = Block(BlockType.LYRICS, lines=[self.right])
        left_block = Block(BlockType.LYRICS, lines=[self.left])
        next_row = Block(BlockType.LYRICS, lines=[Line(coords=rect(0.05, 0.4, 0.60, 0.45))])
        other_column = Block(BlockType.LYRICS, lines=[Line(coords=rect(0.65, 0.2, 0.90, 0.25))])
        page = Page(blocks=[music, right_block, next_row, left_block, other_column])
        self.assertEqual(lyric_lines_for_music_block(page, music, left_block),
                         [(left_block, self.left), (right_block, self.right)])
        self.assertEqual(lyric_lines_for_music_block(page, music, None), [])

    def test_both_lines_of_one_split_block_are_candidates(self):
        music = Block(BlockType.MUSIC, lines=[Line(coords=rect(0.05, 0.1, 0.60, 0.19))])
        lyrics = Block(BlockType.LYRICS, lines=[self.right, self.left])
        page = Page(blocks=[music, lyrics])
        self.assertEqual([l for _, l in lyric_lines_for_music_block(page, music, lyrics)], [self.left, self.right])


class TestChantAndStropheStarts(unittest.TestCase):
    def test_gt_marks_the_first_syllable_of_each_starting_line(self):
        symbols = [note(Pis.SPACE_2), note(Pis.SPACE_3), note(Pis.SPACE_4), note(Pis.SPACE_5)]
        syllables = [Syllable(text=f"s{i}", connection=SyllableConnection.NEW) for i in range(4)]
        first, second = Line(sentence=Sentence(syllables[:2])), Line(sentence=Sentence(syllables[2:]))
        lyrics = Block(BlockType.LYRICS, lines=[first, second])
        music = Block(BlockType.MUSIC, lines=[Line(symbols=symbols)])
        connections = [Connection(music, lyrics, [SyllableConnector(s, n) for s, n in zip(syllables, symbols)])]

        second.start_type = LineStartType.CHANT
        text = process_single_block(None, music, connections).text
        self.assertEqual(text.count('[NEW_CHANT]'), 1)
        self.assertIn(' [NEW_CHANT] s2[', text)

        first.start_type = LineStartType.STROPHE
        text = process_single_block(None, music, connections).text
        self.assertTrue(text.startswith('[NEW_STROPHE] s0['))
        self.assertIn(' [NEW_CHANT] s2[', text)
        self.assertEqual([c.start for c in parse_chunks(text)],
                         [LineStartType.STROPHE, LineStartType.NONE, LineStartType.CHANT, LineStartType.NONE])

    def test_markers_set_the_start_of_the_next_syllable_and_keep_symbol_offsets(self):
        text = "*[(clef|c|7)] [NEW_STROPHE] a[(note|0|6|2)]-b[(note|0|7|2)] [NEW_CHANT] c[(note|0|8|2)]"
        offsets = [i for i in range(len(text)) if text.startswith('(', i)]
        locs = {o: (i / 10, 0.5) for i, o in enumerate(offsets)}
        chunks = parse_chunks(text, locs=locs)
        self.assertEqual([(c.text, c.start) for c in chunks], [('', LineStartType.NONE), ('a', LineStartType.STROPHE),
                                                               ('b', LineStartType.NONE), ('c', LineStartType.CHANT)])
        self.assertEqual([l for c in chunks for l in c.locs], [(i / 10, 0.5) for i in range(4)])
        self.assertEqual(chunks[2].connection, SyllableConnection.HIDDEN)

    def test_a_marker_on_the_first_syllable_only_marks_the_line(self):
        chunks = [syllable_at('a', 0.1, LineStartType.STROPHE), syllable_at('b', 0.2)]
        self.assertEqual(plan_line_starts(chunks, 0.05, 0.6, []), (LineStartType.STROPHE, []))
        self.assertEqual(plan_line_starts(chunks[1:], 0.05, 0.6, []), (None, []))

    def test_mid_line_markers_split_only_at_a_drop_capital(self):
        chunks = [syllable_at('a', 0.1), syllable_at('b', 0.2, LineStartType.CHANT), syllable_at('c', 0.3),
                  syllable_at('d', 0.5, LineStartType.STROPHE)]
        self.assertEqual(plan_line_starts(chunks, 0.05, 0.6, []), (None, []))
        # a drop capital between a and b, none between c and d
        own, splits = plan_line_starts(chunks, 0.05, 0.6, [(0.13, 0.17), (0.52, 0.56)])
        self.assertIsNone(own)
        self.assertEqual([(s.index, s.start) for s in splits], [(1, LineStartType.CHANT)])
        self.assertAlmostEqual(splits[0].x, 0.13)

    def test_a_cut_stays_inside_the_line(self):
        # the drop capital and the marked syllable's note lie right of the line
        chunks = [syllable_at('a', 0.5), syllable_at('b', 0.9, LineStartType.CHANT)]
        _, [split] = plan_line_starts(chunks, 0.05, 0.6, [(0.62, 0.7)])
        self.assertTrue(0.5 < split.x < 0.6)

    def test_split_lines_follow_their_source_in_block_and_reading_order(self):
        line, other = Line(coords=rect(0.05, 0.2, 0.60, 0.25)), Line(coords=rect(0.05, 0.4, 0.60, 0.45))
        lyrics = Block(BlockType.LYRICS, lines=[line, other])
        music = Block(BlockType.MUSIC, lines=[Line(coords=rect(0.05, 0.1, 0.60, 0.19))])
        page = Page(blocks=[music, lyrics])
        page.reading_order.reading_order = [line, other]
        chunks = [syllable_at('a', 0.1), syllable_at('b', 0.2, LineStartType.CHANT), syllable_at('c', 0.3)]
        annotations = Annotations(page)
        drop_capital = Block(BlockType.DROP_CAPITAL, coords=rect(0.15, 0.1, 0.18, 0.26))
        results = lyric_line_results(lyrics, line, [(music, c) for c in chunks], annotations, [drop_capital])

        # nothing is written before the result is applied
        self.assertEqual(lyrics.lines, [line, other])
        self.assertEqual(line.aabb.right(), 0.60)
        apply_text_lines(page, results)

        new = lyrics.lines[1]
        self.assertEqual(lyrics.lines, [line, new, other])
        self.assertEqual(page.reading_order.reading_order, [line, new, other])
        self.assertEqual((line.start_type, new.start_type), (LineStartType.NONE, LineStartType.CHANT))
        self.assertEqual((line.sentence.text(), new.sentence.text()), ('a', 'b c'))
        self.assertAlmostEqual(line.aabb.right(), 0.15)
        self.assertAlmostEqual(new.aabb.left(), 0.15)
        self.assertAlmostEqual(new.aabb.right(), 0.60)
        self.assertEqual([sc.syllable for c in annotations.connections for sc in c.syllable_connections],
                         line.sentence.syllables + new.sentence.syllables)
        wire = [t.to_json() for t in results]
        self.assertEqual(set(wire[0]), {'id', 'sentence', 'syllables', 'coords'})
        self.assertEqual((wire[1]['splitFrom'], wire[1]['blockId'], wire[1]['startType']),
                         (line.id, lyrics.id, 'chant'))


class TestRunnablePages(unittest.TestCase):
    @staticmethod
    def page(staff_lines: bool, lyrics: bool):
        sl = StaffLines([StaffLine(rect(0.05, 0.1, 0.6, 0.1))]) if staff_lines else StaffLines()
        blocks = [Block(BlockType.MUSIC, lines=[Line(coords=rect(0.05, 0.1, 0.6, 0.19), staff_lines=sl)])]
        if lyrics:
            blocks.append(Block(BlockType.LYRICS, lines=[Line(coords=rect(0.05, 0.2, 0.6, 0.25))]))
        return SimpleNamespace(pcgts=lambda: SimpleNamespace(page=Page(blocks=blocks)))

    def test_batch_runs_need_staff_lines_and_lyric_regions(self):
        self.assertTrue(End2EndPredictor.runnable(self.page(staff_lines=True, lyrics=True)))
        self.assertFalse(End2EndPredictor.runnable(self.page(staff_lines=False, lyrics=True)))
        self.assertFalse(End2EndPredictor.runnable(self.page(staff_lines=True, lyrics=False)))


class TinyNetwork(End2EndNetwork):
    def __init__(self, vocab_size, localization, loc_feedback=False):
        super().__init__(vocab_size, d_model=8, nhead=2, num_layers=1, loc_feature_dim=4 if localization else None,
                         loc_feedback=loc_feedback)


class TestFineTuningWithAnExtendedCodec(unittest.TestCase):
    def test_rows_follow_their_tokens_and_class_tokens_start_from_their_base(self):
        with tempfile.TemporaryDirectory() as d:
            old_lines = codec_lines(['a', 'b'])
            new_lines = merge_codec_lines(old_lines, codec_lines(['a', 'b', 'c'], class_specs=[CUSTODES]))
            write_codec(os.path.join(d, 'old.txt'), old_lines)
            write_codec(os.path.join(d, 'new.txt'), new_lines)
            old_tok, new_tok = SmartTokenizer(os.path.join(d, 'old.txt')), SmartTokenizer(os.path.join(d, 'new.txt'))

            old = TinyNetwork(len(old_tok), localization=False)   # a model from before the localisation head
            new = TinyNetwork(len(new_tok), localization=True)
            fresh = load_pretrained(new, old.state_dict(), old_tok, new_tok, {'custodes': CUSTODES})

            self.assertTrue(fresh and all(k.startswith('loc_head.') for k in fresh))
            emb_old, emb_new = old.embedding.weight, new.embedding.weight
            for token in ('b', '(note|0|6|1)', '(clef|c|7)'):
                self.assertTrue(torch.equal(emb_new[new_tok.token2idx[token]], emb_old[old_tok.token2idx[token]]))
            self.assertTrue(torch.equal(new.head.weight[new_tok.token2idx['(clef|@custodes|7)']],
                                        old.head.weight[old_tok.token2idx['(clef|f|7)']]))
            self.assertTrue(torch.equal(new.decoder.layers[0].linear1.weight, old.decoder.layers[0].linear1.weight))


class TestIncrementalDecoding(unittest.TestCase):
    def test_cached_steps_match_the_full_causal_decoder(self):
        from omr.steps.end2end.networks import EncodedImage
        torch.manual_seed(0)
        net = TinyNetwork(20, localization=False).eval()
        # the last two memory entries are image padding
        padding = torch.tensor([[False, False, False, False, True, True]])
        enc = EncodedImage(torch.rand(1, 6, 8), padding, None, None, 32)
        tokens = torch.tensor([[1, 5, 7, 9, 4]])
        with torch.no_grad():
            full = net.decode(tokens, enc)
            cache = net.start_decoding(enc)
            steps = torch.cat([net.decode_next(tokens[:, i], cache, enc) for i in range(tokens.shape[1])], dim=1)
        self.assertTrue(torch.allclose(full, steps, atol=1e-5))

    def test_location_feedback_sees_the_latest_symbol_before_each_position(self):
        from omr.steps.end2end.networks import EncodedImage, previous_symbol_locations
        torch.manual_seed(0)
        net = TinyNetwork(20, localization=True, loc_feedback=True).eval()
        enc = EncodedImage(torch.rand(1, 6, 8), None, None, None, 32)
        tokens = torch.tensor([[1, 5, 7, 9, 4, 6]])
        nan = float('nan')
        # tokens 5 and 4 are symbols located at x = 0.3 and x = 1.2; the others carry no location
        locs = torch.tensor([[[nan, nan], [0.3, 0.5], [nan, nan], [nan, nan], [1.2, 0.4], [nan, nan]]])
        with torch.no_grad():
            full = net.decode(tokens, enc, previous_symbol_locations(locs))
            # inference: a step sees the location of the latest symbol decoded before it
            cache, latest, steps = net.start_decoding(enc), torch.full((1, 2), nan), []
            for i in range(tokens.shape[1]):
                steps.append(net.decode_next(tokens[:, i], cache, enc, latest))
                if not torch.isnan(locs[0, i, 0]):
                    latest = locs[:, i]
            without_feedback = net.decode(tokens, enc, torch.full_like(locs, nan))
        self.assertTrue(torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5))
        # positions 0 and 1 have no symbol before them, the others do
        self.assertTrue(torch.allclose(full[:, :2], without_feedback[:, :2], atol=1e-5))
        self.assertFalse(torch.allclose(full[:, 2:], without_feedback[:, 2:], atol=1e-3))


if __name__ == '__main__':
    unittest.main()
