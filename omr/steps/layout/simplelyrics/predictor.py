from database.model import Model, MetaId
from omr.steps.algorithmtypes import AlgorithmTypes
from omr.steps.layout.drop_capitals.predictor import DropCapitalPredictor
from omr.steps.layout.predictor import LayoutAnalysisPredictor, PredictionType, PredictionResult, \
    PredictionCallback, AlgorithmPredictorSettings, FinalPredictionResult, IdCoordsPair
from typing import List, Optional
from database.file_formats.pcgts import PcGts, BlockType, Coords, Line, Rect, Point, Size
import numpy as np
from omr.steps.layout.lyricsbbs.meta import Meta
from omr.steps.step import Step


class Predictor(LayoutAnalysisPredictor):
    @staticmethod
    def meta():
        return Meta

    def __init__(self, settings: AlgorithmPredictorSettings):
        super().__init__(settings)
        meta = Step.meta(AlgorithmTypes.LAYOUT_SIMPLE_DROP_CAPITAL)
        from ommr4all.settings import BASE_DIR
        model = Model(MetaId.from_custom_path(BASE_DIR + '/internal_storage/default_models/french14/layout_drop_capital/', meta.type()))
        print(model.path)
        settings = AlgorithmPredictorSettings(
            model=model,
        )
        #settings.params.ctcDecoder.params.type = CTCDecoderParams.CTC_DEFAULT
        self.drop_capital: DropCapitalPredictor = meta.create_predictor(settings)
    def _predict_single(self, pcgts_file: PcGts) -> FinalPredictionResult:
        mls_in_cols = pcgts_file.page.all_music_lines_in_columns()
        music_coords: List[IdCoordsPair] = []
        lyric_coords: List[IdCoordsPair] = []
        for mls in mls_in_cols:
            mls = [ml for ml in mls if len(ml.staff_lines) > 0]
            mls.sort(key=lambda ml: ml.center_y())
            pad = pcgts_file.page.avg_staff_line_distance() / 2
            pad_tup = (0, pad)
            for ml in mls:
                ml.staff_lines.sort()

            for m in mls:
                music_coords.append(IdCoordsPair(Coords(np.concatenate(
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
                )), m.id))

            avg_staff_distance = np.mean([m2.staff_lines[0].center_y() - m1.staff_lines[-1].center_y() for m1, m2 in zip(mls[:-1], mls[1:])])
            for m1, m2 in zip(mls[:-1], mls[1:]):
                top_l = m1.staff_lines[-1]
                bot_l = m2.staff_lines[0]
                bot_points = bot_l.coords.points[np.where((top_l.coords.points[0][0] < bot_l.coords.points[:, 0]) & (bot_l.coords.points[:, 0] < top_l.coords.points[-1][0]))]
                bot_points = np.concatenate([
                    [(top_l.coords.points[0][0], bot_l.interpolate_y(top_l.coords.points[0][0]))],
                    bot_points,
                    [(top_l.coords.points[-1][0], bot_l.interpolate_y(top_l.coords.points[-1][0]))],
                ])
                lyric_coords.append(IdCoordsPair(Coords(np.concatenate([
                    top_l.coords.points + pad_tup,
                    [top_l.coords.points[-1] + (0, bot_points[-1][1] - top_l.coords.interpolate_y(bot_points[-1][0])) - pad_tup],
                    bot_points[::-1] - pad_tup,
                    [top_l.coords.points[0] + (0, bot_points[0][1] - top_l.coords.interpolate_y(bot_points[0][0])) - pad_tup],
                ], axis=0))))

            top_l = mls[-1].staff_lines[-1]
            lyric_coords.append(IdCoordsPair(Coords(np.concatenate((
                top_l.coords.points + pad_tup,
                top_l.coords.points[::-1] + (0, avg_staff_distance - pad),
            ), axis=0))))
        drop_capital = True
        drop_capital_blocks = []
        if drop_capital:
            res =list(self.drop_capital._predict([pcgts_file]))[0]
            for x in res.blocks.get(BlockType.DROP_CAPITAL):
                drop_capital_blocks.append(IdCoordsPair(x))
            pass
        return FinalPredictionResult(
            blocks={
                BlockType.LYRICS: lyric_coords,
                BlockType.MUSIC: music_coords,
                BlockType.DROP_CAPITAL: drop_capital_blocks,

            },
            pcgts=pcgts_file,
        )

    def _predict(self, pcgts_files: List[PcGts], callback: Optional[PredictionCallback] = None) -> PredictionType:
        for i, r in enumerate(map(self._predict_single, pcgts_files)):
            if callback:
                callback.progress_updated(i / len(pcgts_files))
            yield r

