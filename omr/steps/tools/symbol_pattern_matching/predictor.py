from typing import List

from database import DatabasePage
from database.file_formats.pcgts import Page, SymbolType
from omr.steps.algorithm import AlgorithmPredictionResult, AlgorithmPredictor, AlgorithmPredictionResultGenerator, \
    PredictionCallback, AlgorithmMeta
from omr.steps.tools.symbol_pattern_matching.meta import Meta
from loguru import logger

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

from typing import List, Dict, Any, Type, Optional, Generator
from dataclasses import dataclass

@dataclass
class PatternMatch:
    line_id: str
    pattern: List[int]
    count: int

from dataclasses import dataclass
from typing import List, Dict, Any, Type, Optional, Generator

@dataclass
class PatternMatch:
    line_id: str
    pattern: List[int]
    count: int
    occurrences_coords: List[List['Point']]

    def get_aabbs(self) -> List[Dict[str, float]]:
        aabbs = []
        for points in self.occurrences_coords:
            if not points: continue
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            aabbs.append({
                "x": min(xs), "y": min(ys),
                "w": max(xs) - min(xs), "h": max(ys) - min(ys)
            })
        return aabbs

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

        patterns_to_search: List[List[int]] = getattr(self.params, 'patterns', [])
        logger.info(patterns_to_search)
        logger.info(self.params)
        for i, db_page in enumerate(pages):
            logger.info(db_page.page)

            page_data = db_page.pcgts().page
            page_data.update_note_names()

            page_matches = []
            page_total = 0

            for line in page_data.all_music_lines():
                notes = [s for s in line.symbols if s.symbol_type == SymbolType.NOTE]
                line_intervals = line.get_relative_pitch_sequence()

                for pattern in patterns_to_search:

                    n_pattern = len(pattern)
                    if n_pattern == 0 or len(line_intervals) < n_pattern + 1:
                        continue

                    occurrences_points = []

                    for j in range(1, len(line_intervals) - n_pattern + 1):
                        window = line_intervals[j: j + n_pattern]

                        if window == pattern:
                            match_notes = notes[j - 1: j + n_pattern]
                            occurrences_points.append([n.coord for n in match_notes])

                    if occurrences_points:
                        match_count = len(occurrences_points)
                        page_matches.append(PatternMatch(
                            line_id=line.id,
                            pattern=pattern,
                            count=match_count,
                            occurrences_coords=occurrences_points
                        ))
                        page_total += match_count

            if page_total > 0:

                yield MelodicPatternResult(
                    page_id=db_page.page,
                    matches=page_matches,
                    total_count=page_total
                )