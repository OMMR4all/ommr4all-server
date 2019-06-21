from omr.layout.predictor import LayoutAnalysisPredictor, LayoutPredictorParameters, PredictionType, PredictionResult \
    , LayoutAnalysisPredictorCallback
from typing import List, Optional
from database.file_formats.pcgts import PcGts, TextRegionType, Coords, MusicLine, Rect, Point, Size
import numpy as np


class LyricsBBSLayoutAnalysisPredictor(LayoutAnalysisPredictor):
    def __init__(self, params: LayoutPredictorParameters):
        super().__init__(params)

    def _predict_single(self, pcgts_file: PcGts) -> PredictionResult:
        mls = pcgts_file.page.all_music_lines()
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
            text_regions={
                TextRegionType.LYRICS: [aabb.to_coords() for aabb in lyrics_aabbs]
            },
            music_regions=music_coords
        )

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[LayoutAnalysisPredictorCallback] = None) -> PredictionType:
        for i, r in enumerate(map(self._predict_single, pcgts_files)):
            if callback:
                callback.progress_updated(i / len(pcgts_files))
            yield r

