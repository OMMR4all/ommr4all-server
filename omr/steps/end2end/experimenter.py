"""Evaluation of the end-to-end algorithms with the symbol detection metrics.

Run through the generic runner, e.g.::

    python omr/experimenter/runner.py --type END2END_CONVNEXT --train <books> --test <books> \
        --cross_folds 1 --n_epoch 100 --early_stopping_max_keep 5 --ignore_gapped_symbols

Reports what the symbol experimenter reports (detection precision / recall / F1 with the
distance threshold ``--symbol_detected_min_distance``, hSAR = "Sequence", dSAR = "Sequence NC",
the diff table) plus the localisation error of the symbols that the sequence alignment
matches, in staff spaces, and the character error rate of the predicted lyrics. ``--ignore_gapped_symbols``
should be set: the codec folds gaped into neume-start connections, so the network cannot predict them.
"""
import copy
import re
from difflib import SequenceMatcher
from typing import List, NamedTuple, Tuple

import edlib
import numpy as np
from prettytable import PrettyTable

from database import DatabaseBook
from database.file_formats.pcgts import PcGts, MusicSymbol, SymbolType
from omr.steps.symboldetection.evaluator import Counts_Types, Clef_Error_Types
from omr.steps.symboldetection.experimenter import SymbolsExperimenter

from .predictor import Chunk, apply_text_lines, assign_symbol_coords


class BlockInfo(NamedTuple):
    staff_space: float   # page units
    # the predicted symbols placed evenly per syllable, as models without localisation do
    evenly_placed: List[MusicSymbol]


class End2EndExtras(NamedTuple):
    blocks: List[BlockInfo]
    # `Sentence.text()` of GT and prediction for every lyric line the predictor wrote
    lyrics: List[Tuple[str, str]]


def evenly_placed(block) -> List[MusicSymbol]:
    chunks = [Chunk(c.text, c.connection, [copy.copy(s) for s in c.symbols], [None] * len(c.symbols))
              for c in block.chunks]
    return assign_symbol_coords(chunks, block.music_line, None, None)


def alignment_label(s: MusicSymbol):
    sub = {SymbolType.NOTE: s.note_type, SymbolType.CLEF: s.clef_type, SymbolType.ACCID: s.accid_type}.get(s.symbol_type)
    return s.symbol_type, sub, s.position_in_staff, s.symbol_class


def with_note_names(symbols: List[MusicSymbol]) -> List[MusicSymbol]:
    """Pitches the predicted symbols by their preceding clef, as `Page.update_note_names`
    does on storing, so that the melody accuracy is meaningful."""
    clef = None
    for s in symbols:
        if s.symbol_type == SymbolType.CLEF:
            clef = s
        elif clef is not None:
            s.update_note_name(clef)
    return symbols


def localisation_errors(gt: List[MusicSymbol], pred: List[MusicSymbol], staff_space: float):
    """(dx, dy, d) of every symbol pair the sequence alignment matches; dx (signed, prediction
    minus GT) and dy in staff spaces, d in page units."""
    sm = SequenceMatcher(a=list(map(alignment_label, gt)), b=list(map(alignment_label, pred)), autojunk=False)
    errors = []
    for block in sm.get_matching_blocks():
        for i in range(block.size):
            g, p = gt[block.a + i].coord, pred[block.b + i].coord
            errors.append(((p.x - g.x) / staff_space, abs(p.y - g.y) / staff_space,
                           float(np.hypot(p.x - g.x, p.y - g.y))))
    return errors


def localisation_summary(errors: List, detection_distance: float) -> np.ndarray:
    """(aligned symbols, median / P90 / mean |dx|, median |dy|, share within the detection distance,
    mean signed dx). The signed mean exposes a systematic shift of all symbols to one side."""
    errors = np.array(errors).reshape(-1, 3)
    if len(errors) == 0:
        return np.array([0] + [np.nan] * 6)
    abs_dx = np.abs(errors[:, 0])
    return np.array([len(errors), *np.percentile(abs_dx, [50, 90]), abs_dx.mean(),
                     np.median(errors[:, 1]), (errors[:, 2] <= detection_distance).mean(), errors[:, 0].mean()])


# medieval orthography varies between the books (Pa writes ſ, u for v, capitals and punctuation; the
# training books mostly do not), so the normalised CER compares lower-case letters with ſ = s, v = u, j = i
ORTHOGRAPHY = str.maketrans({'ſ': 's', 'v': 'u', 'j': 'i'})
NON_LETTERS = re.compile(r'[^a-z]')


def normalise_lyrics(text: str) -> str:
    return NON_LETTERS.sub('', text.lower().translate(ORTHOGRAPHY))


def lyrics_summary(pairs: List) -> np.ndarray:
    """(GT characters, CER, normalised CER) over (gt, pred) `Sentence.text()` pairs. The raw CER
    compares the texts as stored, i.e. wrong word/syllable boundaries (' ', '-', '~') count as
    errors too; the normalised CER compares only the letters (`normalise_lyrics`)."""
    def cer(gts, preds):
        n = sum(map(len, gts))
        errors = sum(edlib.align(p, g)['editDistance'] for g, p in zip(gts, preds))
        return errors / n if n else np.nan
    gts, preds = [g for g, _ in pairs], [p for _, p in pairs]
    return np.array([sum(map(len, gts)), cer(gts, preds),
                     cer(list(map(normalise_lyrics, gts)), list(map(normalise_lyrics, preds)))])


class End2EndExperimenter(SymbolsExperimenter):
    def extract_gt_prediction(self, full_predictions):
        blocks = [b for r in full_predictions for b in r.blocks]
        gt = [b.music_line.symbols for b in blocks]
        pred = [with_note_names(b.symbols) for b in blocks]
        info = [BlockInfo(b.music_line.staff_lines.avg_line_distance(default=0.01), evenly_placed(b)) for b in blocks]
        # at this point the lines still carry the GT: the prediction is only written by `store_to_page`.
        # A line split by a predicted chant or strophe start is compared as a whole.
        lyrics = []
        for r in full_predictions:
            by_source = {}
            for t in r.text_lines:
                if t.split_from is None:
                    by_source[t.line.id] = (t.line.sentence.text(), [t.sentence.text()])
                else:
                    by_source[t.split_from][1].append(t.sentence.text())
            lyrics += [(text, ' '.join(parts)) for text, parts in by_source.values()]
        return (gt, pred), End2EndExtras(info, lyrics)

    def error_type_counts(self, full_predictions, evaluation_params):
        # the pixel classifier specific error analysis does not apply
        return Counts_Types(), Clef_Error_Types()

    def evaluate(self, predictions, evaluation_params):
        (gt_symbols, pred_symbols), extras = predictions
        info = extras.blocks
        d = evaluation_params.symbol_detected_min_distance
        loc = np.stack([
            localisation_summary(sum([localisation_errors(g, p, l.staff_space)
                                      for g, p, l in zip(gt_symbols, pred_symbols, info)], []), d),
            localisation_summary(sum([localisation_errors(g, l.evenly_placed, l.staff_space)
                                      for g, l in zip(gt_symbols, info)], []), d),
        ])
        lyrics = lyrics_summary(extras.lyrics)
        result = super().evaluate(predictions, evaluation_params)
        self.fold_log.debug("\n" + self.localisation_table([loc]).get_string())
        self.fold_log.debug("\n" + self.lyrics_table([lyrics]).get_string())
        return (*result, loc, lyrics)

    @staticmethod
    def localisation_table(locs) -> PrettyTable:
        """`locs` (folds, 2, 7): row 0 the predicted positions, row 1 the even placement baseline."""
        mean = np.array(locs).mean(axis=0)
        at = PrettyTable(["Positions", "Aligned symbols", "Median |dx| [staff space]", "P90 |dx|", "Mean |dx|",
                          "Median |dy| [staff space]", "Within detection distance [%]", "Mean dx (signed)"])
        for name, row in zip(["predicted", "evenly placed (no localisation)"], mean):
            at.add_row([name] + list(row[:5]) + [row[5] * 100, row[6]])
        return at

    @staticmethod
    def lyrics_table(lyrics) -> PrettyTable:
        """`lyrics` (folds, 3): GT characters, CER, normalised CER."""
        n, cer, cer_norm = np.array(lyrics).mean(axis=0)
        at = PrettyTable(["GT lyric characters", "Lyrics CER [%]", "Lyrics CER normalised [%]"])
        at.add_row([int(n), cer * 100, cer_norm * 100])
        return at

    @classmethod
    def print_results(cls, args, results, log):
        super().print_results(args, results, log)
        locs = [r[9] for r in results if r is not None]
        if locs:
            log.info("Localisation (mean over folds):\n" + cls.localisation_table(locs).get_string())
        lyrics = [r[10] for r in results if r is not None]
        if lyrics:
            log.info("Lyrics (mean over folds):\n" + cls.lyrics_table(lyrics).get_string())

    def output_prediction_to_book(self, pred_book: DatabaseBook, output_pcgts: List[PcGts], predictions):
        """Writes what the predictor stores on a page (`PredictionResult.store_to_page`): symbols,
        lyrics, lyric line splits and start types, and syllable->note connections. The copied pages
        keep all ids, so the predicted connections, which serialise by id, refer to the copies' regions and lines."""
        by_page = {p.page.location.page: p for p in output_pcgts}
        for o_pcgts in output_pcgts:
            o_pcgts.page.annotations.connections = []
            o_pcgts.page.comments.comments = []
            for ml in o_pcgts.page.all_music_lines():
                ml.symbols = []
        for r in predictions:
            page = by_page[r.dataset_page.page].page
            for b in r.blocks:
                page.music_line_by_id(b.music_line.id).symbols = b.symbols
            apply_text_lines(page, r.text_lines)
            page.annotations.connections = list(r.annotations.connections)
            page.update_note_names()

    def output_debug_images(self, predictions, params=None):
        raise NotImplementedError("Debug images are not available for end2end models, inspect the "
                                  "predictions in the editor with --output_book instead")
