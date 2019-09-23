from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings
from typing import List, Optional
from database.file_formats.pcgts import PcGts, BlockType, Coords, Line, Rect, Point, Size
import numpy as np
from omr.steps.layout.lyricsbbs.meta import Meta


class Predictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)

    def _predict_single(self, pcgts_file: PcGts) -> PredictionResult:
        mls = pcgts_file.page.all_music_lines()
        mls = [ml for ml in mls if len(ml.staff_lines) > 0]
        mls.sort(key=lambda ml: ml.center_y())
        pad = pcgts_file.page.avg_staff_line_distance() / 2
        pad_tup = (0, pad)
        for ml in mls:
            ml.staff_lines.sort()

        music_coords = []
        for m in mls:
            music_coords.append(Coords(np.concatenate(
                [
                    m.staff_lines[0].coords.points - pad_tup,
                    [
                        c.coords.points[-1] for c in m.staff_lines[1:-1]
                    ],
                    m.staff_lines[-1].coords.points[::-1] + pad_tup,
                    [
                        c.coords.points[0] for c in reversed(m.staff_lines[1:-1])
                    ],
                ]
            )))

        lyric_coords = []
        avg_staff_distance = np.mean([m2.staff_lines[0].center_y() - m1.staff_lines[-1].center_y() for m1, m2 in zip(mls[:-1], mls[1:])])
        for m1, m2 in zip(mls[:-1], mls[1:]):
            top_l = m1.staff_lines[-1]
            bot_l = m2.staff_lines[0]
            lyric_coords.append(Coords(np.concatenate([
                top_l.coords.points + pad_tup, bot_l.coords.points[::-1] - pad_tup], axis=0)))

        top_l = mls[-1].staff_lines[-1]
        lyric_coords.append(Coords(np.concatenate((
            top_l.coords.points + pad_tup,
            top_l.coords.points[::-1] + (0, avg_staff_distance - pad),
        ), axis=0)))

        return PredictionResult(
            blocks={
                BlockType.LYRICS: lyric_coords,
                BlockType.MUSIC: music_coords,
            },
        )

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        for i, r in enumerate(map(self._predict_single, pcgts_files)):
            if callback:
                callback.progress_updated(i / len(pcgts_files))
            yield r

