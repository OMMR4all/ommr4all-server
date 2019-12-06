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
        mls_cols = pcgts_file.page.all_music_lines_in_columns()
        for mls in mls_cols:
            mls.sort(key=lambda ml: ml.center_y())
            ids_aabbs = [(ml.id, ml.staff_lines.aabb()) for ml in mls]
            music_coords = [aabb.to_coords() for _, aabb in ids_aabbs]
            ids_aabbs.sort(key=lambda id_aabb: id_aabb[1].bottom())
            lyrics_aabbs = []
            if len(ids_aabbs) > 0:
                for i in range(len(ids_aabbs) - 1):
                    _, top = ids_aabbs[i]
                    _, bot = ids_aabbs[i + 1]
                    lyrics_aabbs.append(Rect(Point(top.left(), top.bottom()), Size(top.size.w, bot.top() - top.bottom())))

                if len(lyrics_aabbs) > 0:
                    avg_h = np.mean([a.size.h for a in lyrics_aabbs])
                else:
                    avg_h = np.mean([a.size.h for _, a in ids_aabbs])

                _, last = ids_aabbs[-1]
                lyrics_aabbs.append(Rect(Point(last.left(), last.bottom()), Size(last.size.w, avg_h)))

        return PredictionResult(
            blocks={
                BlockType.LYRICS: [aabb.to_coords() for aabb in lyrics_aabbs],
                BlockType.MUSIC: music_coords
            },
        )

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        for i, r in enumerate(map(self._predict_single, pcgts_files)):
            if callback:
                callback.progress_updated(i / len(pcgts_files))
            yield r

