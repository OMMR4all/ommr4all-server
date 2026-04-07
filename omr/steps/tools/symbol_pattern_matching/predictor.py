import os
from database import DatabasePage, DatabaseBook
from database.file_formats.pcgts import Page, SymbolType, PageScaleReference, NoteName, GraphicalConnectionType
from database.file_formats.performance.pageprogress import Locks
from omr.steps.algorithm import AlgorithmPredictionResult, AlgorithmPredictor, AlgorithmPredictionResultGenerator, \
    PredictionCallback, AlgorithmMeta
from omr.steps.algorithmpreditorparams import AlgorithmPredictorParams, AlgorithmPredictorSettings
from omr.steps.tools.symbol_pattern_matching.meta import Meta
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Type, Optional, Generator

class MusicPatternSearch:
    def __init__(self, patterns: List[List[int]]):
        self.patterns = patterns

    def search_dataset(self, pages: List['Page'], group_by="page"):

        results = []

        for page in pages:
            if group_by == "page":
                count = page.get_pattern_stats(self.patterns, granular=False)
                if count > 0:
                    results.append((f"Page ID: {page.p_id}", count))
            else:
                line_stats = page.get_pattern_stats(self.patterns, granular=True)
                for line_id, count in line_stats:
                    results.append((f"Line ID: {line_id} (Page: {page.p_id})", count))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


@dataclass
class PatternMatch:
    line_id: str
    pattern: List[int]
    count: int
    occurrences_coords: List[List['Point']]
    occurrences_aabbs: List[Dict[str, float]]

    def get_aabbs(self) -> List[Dict[str, float]]:
        return self.occurrences_aabbs

class MelodicPatternResult(AlgorithmPredictionResult):
    def __init__(self, page_id: str, matches: List[PatternMatch], total_count: int):
        self.page_id = page_id
        self.matches = matches
        self.total_count = total_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "total_count": self.total_count,
            "matches": [
                {
                    "line_id": m.line_id,
                    "pattern": m.pattern,
                    "count": m.count,
                    "boxes": m.get_aabbs()
                } for m in self.matches
            ]
        }

    def store_to_page(self):
        pass


class MelodicPatternPredictor(AlgorithmPredictor):

    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        return Meta

    @classmethod
    def unprocessed(cls, page: 'DatabasePage') -> bool:
        try:
            return len(page.pcgts().page.music_blocks()) > 0
        except Exception:
            return False

    def predict(self, pages: List['DatabasePage'],
                callback: Optional['PredictionCallback'] = None) -> AlgorithmPredictionResultGenerator:

        patterns_to_search: List[List[Any]] = getattr(self.params, 'patterns', [])
        syllable_only: bool = getattr(self.params, 'syllable_only', True)

        for i, db_page in enumerate(pages):
            page_data = db_page.pcgts().page
            page_data.update_note_names()

            staff_space = page_data.avg_staff_line_distance()
            min_pad_x = staff_space * 0.4
            max_pad_x = staff_space * 1.5
            pad_y = staff_space * 2.0

            note_to_syllable = {}
            if syllable_only:
                for connection in page_data.annotations.connections:
                    for sc in connection.syllable_connections:
                        note_to_syllable[sc.note.id] = sc.syllable.id

            page_matches = []
            page_total = 0

            for line in page_data.all_music_lines():
                notes = [s for s in line.symbols if
                         s.symbol_type == SymbolType.NOTE and s.note_name != NoteName.UNDEFINED]
                if not notes:
                    continue

                if syllable_only:
                    chunks = []
                    current_chunk = []
                    for note in notes:
                        if note.id in note_to_syllable:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = [note]
                        elif current_chunk:
                            current_chunk.append(note)
                    if current_chunk:
                        chunks.append(current_chunk)
                else:
                    chunks = [notes]

                for chunk_notes in chunks:
                    if len(chunk_notes) < 2:
                        continue

                    abs_pitches = [(n.octave * 7) + n.note_name.value for n in chunk_notes]
                    rel_pitches = [0] + [abs_pitches[k] - abs_pitches[k - 1] for k in range(1, len(abs_pitches))]

                    # We ALWAYS build the sequence with connections now
                    chunk_sequence = []
                    for k in range(len(chunk_notes)):
                        pitch = rel_pitches[k]
                        conn_val = chunk_notes[k].graphical_connection
                        is_looped = 1 if conn_val == GraphicalConnectionType.LOOPED else 0
                        chunk_sequence.append([pitch, is_looped])

                    for pattern in patterns_to_search:
                        n_pattern = len(pattern)
                        if n_pattern == 0 or len(chunk_sequence) < n_pattern + 1:
                            continue

                        occurrences_points = []
                        occurrences_aabbs = []

                        for j in range(1, len(chunk_sequence) - n_pattern + 1):
                            window = chunk_sequence[j: j + n_pattern]

                            is_match = True
                            for w, p in zip(window, pattern):
                                w_pitch, w_conn = w
                                p_pitch, p_conn = p

                                if w_pitch != p_pitch:
                                    is_match = False
                                    break

                                if p_conn is not None and w_conn != p_conn:
                                    is_match = False
                                    break

                            if is_match:
                                match_notes = chunk_notes[j - 1: j + n_pattern]
                                occurrences_points.append([n.coord for n in match_notes])

                                xs = [n.coord.x for n in match_notes]
                                ys = [n.coord.y for n in match_notes]
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)

                                first_note_idx = notes.index(match_notes[0])
                                last_note_idx = notes.index(match_notes[-1])

                                prev_note = notes[first_note_idx - 1] if first_note_idx > 0 else None
                                next_note = notes[last_note_idx + 1] if last_note_idx < len(notes) - 1 else None

                                left_pad = (x_min - prev_note.coord.x) / 2.0 if prev_note else staff_space * 0.6
                                right_pad = (next_note.coord.x - x_max) / 2.0 if next_note else staff_space * 0.6

                                left_pad = max(min_pad_x, min(left_pad, max_pad_x))
                                right_pad = max(min_pad_x, min(right_pad, max_pad_x))

                                occurrences_aabbs.append({
                                    "x": x_min - left_pad,
                                    "y": y_min - pad_y,
                                    "w": (x_max + right_pad) - (x_min - left_pad),
                                    "h": (y_max + pad_y) - (y_min - pad_y)
                                })

                        if occurrences_points:
                            match_count = len(occurrences_points)
                            page_matches.append(PatternMatch(
                                line_id=line.id,
                                pattern=pattern,
                                count=match_count,
                                occurrences_coords=occurrences_points,
                                occurrences_aabbs=occurrences_aabbs
                            ))
                            page_total += match_count

            if page_total > 0:
                yield MelodicPatternResult(
                    page_id=db_page.page,
                    matches=page_matches,
                    total_count=page_total
                )

logging.basicConfig(level=logging.INFO)


def draw_predictions(image_path: str, prediction_results: MelodicPatternResult, ide):
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    ps = PageScaleReference(0)

    for ind, match in enumerate(prediction_results.matches):
        for aabbs in match.get_aabbs():
            x_min = int(ide.page.pcgts().page.page_to_image_scale(aabbs["x"], ps))
            y_min = int(ide.page.pcgts().page.page_to_image_scale(aabbs["y"], ps))
            x_max = int(ide.page.pcgts().page.page_to_image_scale(aabbs["x"] + aabbs["w"], ps))
            y_max = int(ide.page.pcgts().page.page_to_image_scale(aabbs["y"] + aabbs["h"], ps))

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

            color = colors[ind % len(colors)]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)

            label = f"Pattern: {match.pattern}"
            text_y = y_min - 10 if y_min - 10 > 10 else y_min + 20
            cv2.putText(image, label, (x_min, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    from matplotlib import pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()



if __name__ == "__main__":
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    import random
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState
    random.seed(1)
    np.random.seed(1)
    b = DatabaseBook('Graduel_Part_3_gt')
    train_pcgts, val_pcgts = dataset_by_locked_pages(0.8, [LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True)], True, [b])

    pred = MelodicPatternPredictor(AlgorithmPredictorSettings(
        model=Meta.best_model_for_book(b),
        params=AlgorithmPredictorParams(
            patterns=[[(1, 1), (1, 0)]],
            include_graphical_connection=True,
        )
    ))
    pages = [p.page.location for p in train_pcgts[1:4]]

    for t in pred.predict(pages):
        page = next(p for p in pages if p.page == t.page_id)

        ide = page.file("color_original")
        draw_predictions(ide.local_path(), t, ide)
        print(t.to_dict())

